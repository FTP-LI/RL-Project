#include <Arduino.h>
#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include "qmi8658.h"

static int readReg(int reg) {
  if (reg < 0 || reg > 0x60) return -3;
  Wire.beginTransmission((uint8_t)0x6A);
  Wire.write((uint8_t)reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom((uint8_t)0x6A, (size_t)1) != 1) return -2;
  return Wire.read();
}

static MsgPack::str_t initProbe(int waitMs) {
  if (waitMs != 20 && waitMs != 100 && waitMs != 500) return "invalid wait";
  diagnosticResetWaitMs = waitMs;
  const bool ok = initImu();
  const int reset = readReg(0x4D);
  int config[7];
  for (int i = 0; i < 7; ++i) config[i] = readReg(0x02 + i);
  const bool sampled = ok && updateImu();
  char result[300];
  snprintf(result, sizeof(result),
    "{\"init_ok\":%d,\"stage\":%d,\"reset_result\":%d,\"ctrl\":[%d,%d,%d,%d,%d,%d,%d],\"sample_ok\":%d,\"acc\":[%d,%d,%d]}",
    (int)ok, diagnosticStage, reset, config[0], config[1], config[2], config[3], config[4], config[5], config[6],
    (int)sampled, imu.accRaw[0], imu.accRaw[1], imu.accRaw[2]);
  return MsgPack::str_t(result);
}

// Temporary bring-up probe. No servo UART is initialized or written.
static MsgPack::str_t probe(int clockHz) {
  if (clockHz != 100000 && clockHz != 400000) return "invalid clock";
  Wire.setClock(clockHz);
  int tx[2], count[2], who[2];
  for (int i = 0; i < 2; ++i) {
    const uint8_t address = 0x6A + i;
    Wire.beginTransmission(address);
    Wire.write((uint8_t)0x00);
    tx[i] = Wire.endTransmission(false);
    count[i] = 0;
    who[i] = -1;
    if (tx[i] == 0) {
      count[i] = Wire.requestFrom(address, (size_t)1);
      if (count[i] == 1) who[i] = Wire.read();
    }
  }
  char result[240];
  snprintf(result, sizeof(result),
    "{\"clock_hz\":%d,\"0x6A\":{\"write_status\":%d,\"read_count\":%d,\"who_am_i\":%d},\"0x6B\":{\"write_status\":%d,\"read_count\":%d,\"who_am_i\":%d}}",
    clockHz, tx[0], count[0], who[0], tx[1], count[1], who[1]);
  return MsgPack::str_t(result);
}

void setup() {
  Wire.begin();
  while (!Bridge.begin(2000000)) delay(500);
  while (!Bridge.provide("codex_imu_probe", probe)) delay(500);
  while (!Bridge.provide("codex_imu_init", initProbe)) delay(500);
  while (!Bridge.provide("codex_imu_read_reg", readReg)) delay(500);
}

void loop() { delay(1000); }
