/*
 * TEAM SANDI — Smart Sanitary Disposal Unit (ESP32)
 * Sensors:
 *   IR  : E18-D80NK / FC-51 obstacle sensor (digital) or Sharp GP2Y0A21 (analog)
 *   Temp: DS18B20 (OneWire)  — chamber / bin temperature
 *   Load: HX711 + 5 kg load cell — bin fill weight
 *   Power (optional, for energy measurement): INA219 on the 5V rail
 *
 * Streams one CSV line every 100 ms over Serial (115200):
 *   ts_ms,ir,temp_c,weight_g,bus_mv,current_ma
 * The PC side (src/logger.py) windows 32 samples (3.2 s) and runs the model.
 *
 * Libraries: OneWire, DallasTemperature, HX711 (bogde), Adafruit_INA219
 */
#include <OneWire.h>
#include <DallasTemperature.h>
#include <HX711.h>
#include <Wire.h>
#include <Adafruit_INA219.h>

// ---------- Pins ----------
#define PIN_IR_DIGITAL  27     // obstacle sensor OUT (LOW = object)
#define PIN_IR_ANALOG   34     // use if Sharp analog IR is fitted
#define PIN_DS18B20     4
#define PIN_HX_DOUT     16
#define PIN_HX_SCK      17
#define USE_ANALOG_IR   0

// ---------- Calibration ----------
float HX_CAL_FACTOR = -420.0f;  // calibrate: known weight -> adjust until reading matches

OneWire oneWire(PIN_DS18B20);
DallasTemperature ds(&oneWire);
HX711 scale;
Adafruit_INA219 ina;
bool inaOk = false;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_IR_DIGITAL, INPUT_PULLUP);
  ds.begin();
  ds.setWaitForConversion(false);   // non-blocking temperature reads
  ds.setResolution(10);             // 10-bit = ~190 ms conversion, lower energy
  scale.begin(PIN_HX_DOUT, PIN_HX_SCK);
  scale.set_scale(HX_CAL_FACTOR);
  scale.tare();
  inaOk = ina.begin();
  ds.requestTemperatures();
  Serial.println("ts_ms,ir,temp_c,weight_g,bus_mv,current_ma");
}

float lastTemp = 25.0f;
uint32_t lastTempReq = 0;

void loop() {
  static uint32_t next = millis();
  if ((int32_t)(millis() - next) < 0) return;
  next += 100;   // 10 Hz

  float ir;
#if USE_ANALOG_IR
  ir = analogRead(PIN_IR_ANALOG) / 4095.0f;          // 0..1 proximity
#else
  ir = digitalRead(PIN_IR_DIGITAL) == LOW ? 1.0f : 0.0f;
#endif

  if (millis() - lastTempReq > 250) {               // async DS18B20
    float t = ds.getTempCByIndex(0);
    if (t > -100) lastTemp = t;
    ds.requestTemperatures();
    lastTempReq = millis();
  }

  float w = scale.is_ready() ? scale.get_units(1) : NAN;

  float busmv = inaOk ? ina.getBusVoltage_V() * 1000.0f : 0;
  float ma    = inaOk ? ina.getCurrent_mA() : 0;

  Serial.printf("%lu,%.2f,%.2f,%.1f,%.0f,%.1f\n",
                millis(), ir, lastTemp, w, busmv, ma);
}
