#pragma once
/* Runtime-switchable Teleplot telemetry -------------------------------------
 *
 * Prints are grouped into named channels that can be switched on/off over the
 * USB serial port while the controller is running (used by the web dashboard
 * in software/telemetry_dashboard, but plain text so a serial monitor works too):
 *
 *   tlm?               -> reply with the channel catalog
 *   tlm <name> 1|0     -> enable / disable one channel
 *   tlm all 1|0        -> enable / disable every channel
 *   tlm rate <ms>      -> print period (10..1000 ms)
 *
 * The catalog is one line starting with "#tlm " followed by JSON:
 *   #tlm {"rate":20,"ch":[{"n":"lqr_torque","on":1,"v":"LQR-tau-x LQR-tau-y"},...]}
 * It is printed at boot, on "tlm?" and after every change.
 *
 * Commands are read non-blocking from loop(); a partial line costs nothing.
 */
#include <Arduino.h>
#include <string.h>
#include <stdlib.h>

struct TelemetryChannel {
  const char* name;       // short id used in commands: [a-z0-9_]
  const char* variables;  // space separated Teleplot names, shown in the dashboard
  bool enabled;
};

class Telemetry {
 public:
  Telemetry(TelemetryChannel* channels, size_t count, uint32_t periodMs)
      : channels_(channels), count_(count), periodMs_(periodMs) {}

  bool on(size_t id) const { return id < count_ && channels_[id].enabled; }

  // True once per print period; call from loop() before printing.
  bool due(uint32_t now) {
    if ((uint32_t)(now - lastPrintMs_) < periodMs_) return false;
    lastPrintMs_ = now;
    return true;
  }

  // Read pending command bytes (non-blocking) and execute complete lines.
  void poll(Stream& io) {
    while (io.available() > 0) {
      const char c = (char)io.read();
      if (c == '\r') continue;
      if (c == '\n') {
        line_[len_] = '\0';
        if (len_ > 0 && !overflow_) execute(io, line_);
        len_ = 0;
        overflow_ = false;
      } else if (len_ < sizeof(line_) - 1) {
        line_[len_++] = c;
      } else {
        overflow_ = true;
      }
    }
  }

  void printCatalog(Print& out, bool boot = false) const {
    out.print("#tlm {\"rate\":");
    out.print(periodMs_);
    if (boot) out.print(",\"boot\":1");
    out.print(",\"ch\":[");
    for (size_t i = 0; i < count_; ++i) {
      if (i) out.print(',');
      out.print("{\"n\":\"");
      out.print(channels_[i].name);
      out.print("\",\"on\":");
      out.print(channels_[i].enabled ? 1 : 0);
      out.print(",\"v\":\"");
      out.print(channels_[i].variables);
      out.print("\"}");
    }
    out.println("]}");
  }

  // Teleplot line: >name:value
  static void value(Print& out, const char* name, float v, int digits = 4) {
    out.print('>');
    out.print(name);
    out.print(':');
    out.println(v, digits);
  }
  static void value(Print& out, const char* name, long v) {
    out.print('>');
    out.print(name);
    out.print(':');
    out.println(v);
  }

 private:
  void execute(Print& out, char* line) {
    char* save = nullptr;
    char* cmd = strtok_r(line, " \t", &save);
    if (!cmd) return;
    if (strcmp(cmd, "tlm?") == 0) {
      printCatalog(out);
      return;
    }
    if (strncmp(cmd, "tlm", 3) != 0) {
      // Not for us. Silently ignored: when the host opens the port, Linux can
      // briefly echo our own output back to us, which arrives here as garbage.
      return;
    }
    if (strcmp(cmd, "tlm") != 0) {
      out.print("tlm: unknown command ");
      out.println(cmd);
      return;
    }
    char* target = strtok_r(nullptr, " \t", &save);
    char* arg = strtok_r(nullptr, " \t", &save);
    if (!target || strcmp(target, "?") == 0) {
      printCatalog(out);
      return;
    }
    if (!arg) {
      out.println("tlm: missing argument");
      return;
    }
    const long n = strtol(arg, nullptr, 10);
    if (strcmp(target, "rate") == 0) {
      periodMs_ = (uint32_t)constrain(n, 10L, 1000L);
    } else if (strcmp(target, "all") == 0) {
      for (size_t i = 0; i < count_; ++i) channels_[i].enabled = n != 0;
    } else {
      size_t i = 0;
      while (i < count_ && strcmp(channels_[i].name, target) != 0) ++i;
      if (i == count_) {
        out.print("tlm: unknown channel ");
        out.println(target);
        return;
      }
      channels_[i].enabled = n != 0;
    }
    printCatalog(out);
  }

  TelemetryChannel* channels_;
  size_t count_;
  uint32_t periodMs_;
  uint32_t lastPrintMs_ = 0;
  char line_[64];
  size_t len_ = 0;
  bool overflow_ = false;
};
