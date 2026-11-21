"""Validate full meshes, video and visible GLFW with process-local drivers.

Why: CUDA works while the observed renderer used llvmpipe. An 8x8 context
query alone does not demonstrate complete MuJoCo rendering or video output.
Change: select graphics drivers only in four short, sequential child processes.
Use the official static STAND pose and an orbiting camera, with a visible
label. Never step physics, load a policy, change training, resize an existing
window, or inspect the previously denied native application window.

Default is a CPU/file preflight. Actual rendering waits for the current pilot
and its fixed evaluation, holds the project compute lock, and must be launched
from the existing visible terminal. This checks rendering capability; complete
training throughput and the actual passive observer still need a separate test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'configs/experiments/renderer_validation.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def preflight():
    protocol = read(PROTOCOL)
    assert protocol['status'] == 'frozen-before-rendering'
    assert (protocol['width'], protocol['height'], protocol['frames'], protocol['video_fps']) == (960, 720, 120, 30)
    assert protocol['physics_steps'] == protocol['learning_updates'] == 0
    assert [(c['backend'], c['driver']) for c in protocol['cases']] == [
        ('egl', 'default'), ('glfw', 'default'), ('egl', 'd3d12'), ('glfw', 'd3d12')]
    scene = ROOT/protocol['scene']
    tree = ET.parse(scene).getroot()
    assert any(k.get('name') == protocol['pose'] for k in tree.findall('./keyframe/key'))
    robot = scene.parent/tree.find('include').get('file')
    robot_xml = ET.parse(robot).getroot()
    assets = robot.parent/robot_xml.find('compiler').get('meshdir')
    inputs = [scene, robot, *[assets/m.get('file') for m in robot_xml.findall('./asset/mesh')]]
    assert len(inputs) > 10 and all(p.is_file() for p in inputs)
    pilot = ROOT/protocol['required_completed_pilot']
    assert sha(pilot/'protocol.json') == protocol['pilot_protocol_sha256']
    verified = pilot/'verification.json'
    ready = verified.is_file() and read(verified)['status'] == 'passed'
    state = read(ROOT/'project.json')
    paused = state['formal_training']['status'].startswith('paused')
    return protocol, dict(status='inputs-verified', checked_at=datetime.now(timezone.utc).isoformat(),
                         source_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                         inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                         pilot_completed_and_verified=ready, user_training_pause_active=paused,
                         rendering_started=False, graphics_context_created=False,
                         training_speedup_verified=False)


def worker(folder, name):
    """One context in a fresh process; render pixels are never window captures."""
    import signal
    protocol, check = preflight()
    assert check['pilot_completed_and_verified'] and not check['user_training_pause_active']
    folder = folder.resolve()
    assert folder.parent.parent == ROOT/'records/benchmarks'
    case = next(c for c in protocol['cases'] if c['name'] == name)
    batch = read(folder.parent/'batch.json')
    assert batch['source_sha256'] == sha(Path(__file__)) and batch['protocol_sha256'] == sha(PROTOCOL)
    assert check['inputs_sha256'] == read(folder.parent/'preflight.json')['inputs_sha256']
    assert os.environ['MUJOCO_GL'] == case['backend']
    if case['driver'] == 'd3d12':
        assert os.environ['GALLIUM_DRIVER'] == 'd3d12'
        assert os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] == 'NVIDIA'
    # The default OS alarm also terminates a child stuck inside a graphics call.
    signal.alarm(protocol['case_timeout_seconds'])
    import mujoco
    import numpy as np
    from OpenGL import GL
    from PIL import Image
    import imageio_ffmpeg
    width, height = protocol['width'], protocol['height']
    model = mujoco.MjModel.from_xml_path(str(ROOT/protocol['scene']))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, protocol['pose'])
    assert key >= 0 and model.nmesh > 10
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)  # Kinematics only, no integration.
    fixed_qpos = data.qpos.copy()
    context = renderer = writer = window = None
    glfw = None
    frame_hashes, native_size = [], None
    started = time.perf_counter()
    try:
        if case['backend'] == 'egl':
            from mujoco.egl import GLContext
            context = GLContext(width, height)
            context.make_current()
        else:
            import glfw
            assert glfw.init(), 'GLFW initialization failed'
            # Create one new diagnostic window. Never resize/reposition another.
            window = glfw.create_window(width, height, 'XGO-Duck | static renderer validation', None, None)
            assert window, 'GLFW window creation failed'
            glfw.make_context_current(window)
            glfw.swap_interval(1)
            native_size = list(glfw.get_framebuffer_size(window))
        graphics = {k: GL.glGetString(v).decode() for k, v in
                    [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
        if case['driver'] == 'd3d12':
            assert all(word in graphics['renderer'] for word in protocol['expected_hardware_renderer_contains']), graphics
        print('RENDER_CONTEXT:', json.dumps(graphics), flush=True)
        renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer)
        assert renderer.currentBuffer == mujoco.mjtFramebuffer.mjFB_OFFSCREEN
        scene = mujoco.MjvScene(model, maxgeom=10000)
        camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
        camera.lookat[:] = [0, 0, 0.12]
        camera.distance, camera.elevation = 0.65, -15
        viewport = mujoco.MjrRect(0, 0, width, height)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        video = folder/'static-render-validation.mp4'
        writer = imageio_ffmpeg.write_frames(str(video), (width, height), fps=protocol['video_fps'],
                                             codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p')
        writer.send(None)
        for index in range(protocol['frames']):
            tick = time.perf_counter()
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer)
            camera.azimuth = 135+index*0.5
            mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, renderer)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, 'STATIC RENDER VALIDATION\nNo policy / no physics steps', name, renderer)
            mujoco.mjr_readPixels(pixels, None, viewport, renderer)
            rgb = np.ascontiguousarray(pixels[::-1])
            assert float(rgb.std()) > 1 and 0 < float(rgb.mean()) < 255
            frame_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            writer.send(rgb.tobytes())
            if index in (0, protocol['frames']//2, protocol['frames']-1):
                Image.fromarray(rgb).save(folder/f'frame-{index:03d}.png')
            if window:
                assert not glfw.window_should_close(window), 'Diagnostic window closed before completion'
                size = glfw.get_framebuffer_size(window)
                mujoco.mjr_blitBuffer(viewport, mujoco.MjrRect(0, 0, *size), 1, 0, renderer)
                glfw.swap_buffers(window)
                glfw.poll_events()
                time.sleep(max(0, 1/protocol['video_fps']-(time.perf_counter()-tick)))
            if index % 30 == 0:
                print(f'RENDER_FRAME: {name} {index+1}/{protocol["frames"]}', flush=True)
        np.testing.assert_array_equal(data.qpos, fixed_qpos)
        assert data.time == 0 and len(set(frame_hashes)) > protocol['frames']//2
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.free()
        if context is not None:
            context.free()
        if glfw is not None:
            if window:
                glfw.destroy_window(window)
            glfw.terminate()
    decoded = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    metadata = next(decoded)
    count = sum(1 for _ in decoded)
    assert count == protocol['frames'] and tuple(metadata['size']) == (width, height)
    assert metadata['fps'] == protocol['video_fps']
    result = dict(status='passed', case=case, checked_at=datetime.now(timezone.utc).isoformat(),
                  graphics=graphics, mujoco=mujoco.__version__, mesh_count=model.nmesh,
                  frame_hashes=frame_hashes, decoded_frames=count, video_fps=metadata['fps'],
                  video_sha256=sha(video), native_framebuffer_size=native_size,
                  wall_seconds_including_setup_and_decode=time.perf_counter()-started,
                  graphics_cleanup_completed=True, physics_steps=0, learning_updates=0,
                  existing_window_accessed=False, training_speedup_verified=False,
                  scope='Full static robot rendering and video; not actual passive-viewer or training throughput validation')
    write(folder/'result.json', result)
    signal.alarm(0)
    print('RENDER_CASE_COMPLETED:', name, flush=True)


def execute():
    import fcntl
    protocol, check = preflight()
    assert check['pilot_completed_and_verified'], 'Finish the current pilot and fixed evaluation first'
    assert not check['user_training_pause_active'], 'User pause remains active'
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = ROOT/'records/benchmarks'/datetime.now().strftime('renderer_%Y%m%dT%H%M%S')
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copy2(PROTOCOL, folder/'protocol.json')
        shutil.copy2(__file__, folder/'validate_renderer.py')
        write(folder/'preflight.json', check)
        batch = dict(status='running', source_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL), cases=[])
        write(folder/'batch.json', batch)
        try:
            for case in protocol['cases']:
                trial = folder/case['name']; trial.mkdir()
                environment = os.environ.copy()
                keys = ('GALLIUM_DRIVER', 'MESA_D3D12_DEFAULT_ADAPTER_NAME', 'LIBGL_ALWAYS_SOFTWARE',
                        'MESA_LOADER_DRIVER_OVERRIDE', 'PYOPENGL_PLATFORM')
                inherited = {key: environment.pop(key, None) for key in keys}
                environment.update(MUJOCO_GL=case['backend'], PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
                if case['driver'] == 'd3d12':
                    environment.update(GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
                command = [sys.executable, '-u', '-B', str(Path(__file__)), '--worker', str(trial), '--case', case['name']]
                write(trial/'request.json', dict(command=command, inherited_graphics_settings=inherited,
                                                selected_graphics_settings={k: environment.get(k) for k in (*keys, 'MUJOCO_GL')}))
                print('VISIBLE_RENDER_VALIDATION:', case['name'], json.dumps(command), flush=True)
                with (trial/'console.log').open('w', buffering=1) as log:
                    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    try:
                        for line in process.stdout:
                            print(line, end='', flush=True); log.write(line)
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try: process.wait(timeout=10)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
                        raise
                batch['cases'].append(dict(name=case['name'], exit_code=code))
                write(folder/'batch.json', batch)
                assert code == 0 and read(trial/'result.json')['status'] == 'passed', case['name']
            batch['status'] = 'rendering-completed-awaiting-visual-review'
        except BaseException as error:
            batch.update(status='failed', error=repr(error))
            raise
        finally:
            batch['finished_at'] = datetime.now(timezone.utc).isoformat()
            write(folder/'batch.json', batch)
        print('RENDER_VALIDATION_COMPLETED:', folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--worker', type=Path)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.worker:
        assert args.case
        worker(args.worker, args.case)
    elif args.execute:
        execute()
    else:
        _, result = preflight()
        output = ROOT/'records/reports/renderer-validation-preflight.json'
        write(output, result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
