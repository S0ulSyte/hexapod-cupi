# Single-leg stand: sensing BOM

**Target:** one USB cable to a computer for power and force/position logging; no proprietary hub. Motors need separate power and CAN hardware. **Firmware, USB power qualification and calibration remain unfinished. Defer the bench-supply purchase.**

## Parts

Prices checked **8 September 2026**, before tax/shipping. Quantities are for one stand.

| Qty | Part | USD | Purpose |
| --- | --- | ---: | --- |
| 1 | [Phidgets ENC4110, 300 mm](https://www.phidgets.com/?prodid=1106) | 90.00 | Carriage position; mounting kit included |
| 1 kit | [Wishiot **2sets 20kg + HX711**](https://www.amazon.com/dp/B0CRDG24R8) | 11.99 | One force cell plus spare |
| 1 | [Waveshare **WAV_11010 / ADS1256**](https://www.amazon.com/dp/B083WN119J) | 43.19 | Fast load-cell readout |
| 1 | [Pico 2 with headers](https://www.adafruit.com/product/6328) | 7.50 | Counter, timestamps and USB |
| 1 | [Terminal PiCowbell](https://www.adafruit.com/product/5907) | 12.50 | Screw-terminal carrier |
| 1 | [74LVC245](https://www.adafruit.com/product/735) | 1.50 | Encoder 5 V → Pico 3.3 V |
| 1 | [DB9 female breakout](https://www.adafruit.com/product/3122) | 2.95 | Encoder connector |
| 1 | [USB A–micro-B data cable](https://www.adafruit.com/product/592) | 2.95 | Computer connection |
| 1 | [Half-size Perma-Proto](https://www.adafruit.com/product/1609) | 4.50 | Soldered interface circuit |
| 1 | [Ribbon wire, 1 m](https://www.adafruit.com/product/6182) | 2.50 | Internal wiring |
| 1 pack | [Female/male jumpers](https://www.adafruit.com/product/1954) | 1.95 | ADC header connections |
| 10 | [100 nF X7R capacitors](https://www.digikey.com/en/products/detail/vishay-beyschlag-draloric-bc-components/K104K15X7RF5TL2/286538) | 1.70 | Decoupling and spares |
| | **Electronics subtotal** | **183.23** | |

Add **$40–70** for enclosure, standoffs, terminals, strain relief, home switch and shielded sensor extension if needed. USB power-conditioning parts, machined fixtures, tools and motor hardware are **not included**.

Recheck prices and availability before purchasing. The original dated selections
remain in Git; the team has not qualified the assembled sensing chain.

## Why this encoder

The ENC4110 has documented **5 µm resolution, 1 m/s maximum speed and 300 mm travel**. The [linked potentiometer](https://www.amazon.com/dp/B0DLGVRCWM?th=1) has 150 mm travel and no usable accuracy specification. The encoder needs a unique home reference; verify distance/count and installed accuracy.

## Wiring essentials

**Encoder → level converter → Pico; load cell → ADS1256 → Pico → USB.**

| Connection | Wiring / setting |
| --- | --- |
| Encoder DB9 | 2 ground; 4 shield; 6 A; 7 +5 V; 8 B; 9 index. Not RS-232. |
| 74LVC245 | Supply 3.3 V; 100 nF decoupling; DIR high, OE low for A→B conversion. Define unused inputs. |
| Pico encoder pins | GP10/11 A/B; GP12 index; GP13 home switch with pull-up. |
| Wishiot bridge | Confirm supplied colors: red excitation+, black ground, green AD2, white AD3. |
| ADS1256 | AD2–AD3 differential; buffer on; gain 64; 2,000 samples/s; onboard 2.5 V reference. VREF jumper serves the DAC. |
| Pico SPI0 pins | GP18 clock; GP19 MOSI; GP16 MISO; GP17 CS; GP20 DRDY; GP21 reset. Hold SYNC/PDWN high; keep unused DAC deselected. |
| Power | ADC needs separate 3.3 V digital and nominal 5 V analog rails. Join grounds; no 5 V into Pico GPIO and no supply backfeed into USB. |

Use the [Waveshare schematic](https://files.waveshare.com/upload/2/29/High-Precision-AD-DA-board.pdf) for header connections; these are wired boards, not a Pico-compatible stack. Keep test/DAC outputs disconnected from force inputs. **The included HX711 is for slow tests only (10/80 samples/s).**

## Before recording useful dynamics

1. **Qualify USB power:** measure startup/running current, loaded voltage and motor-on noise. USB VBUS is not precision 5 V; the cell specifies 5 V minimum excitation. Regulation/filtering is still to be selected. Stay within granted USB current and check suspend behavior.
2. **Build the fixture:** confirm stroke fits, support only the cell's fixed end, leave flex clearance, and add carriage/end/overload stops. Include a stiff foot plate and strain relief without bypassing the measured load path.
3. **Implement logging:** timestamp ADC DRDY and encoder counts on the Pico; retain raw values and dropped-sample counters. Characterize ADC delay and CAN timing separately. This ADC wiring is not ratiometric—measure excitation stability.
4. **Calibrate:** use known loads at intended foot positions, an independent position reference, and repeated homing. Check hysteresis, rail friction and plate response. The 20 kg rating includes tare; impact/eccentric-load capability is unverified. Sampling rate is not mechanical bandwidth.

Use lab power and measurement tools for commissioning. Final motor supply, regenerative-energy handling, stop circuit, XT30 2+2 harness and CAN interface remain outside this sensing BOM.

## Actuator calibration (simulator against the stand)

`src/locomotion/sysid/` measures how the lf leg's RS05 joints respond to the policy's own commands and
fits the simulator to match, actuation delay included.

| Step | Where | Command |
| --- | --- | --- |
| Free the bus | stand computer | stop `leg_app.py`, `leg_jog.py` and every other leg program: one program on `vcan0` at a time |
| Calibrate limits | stand computer | `leg_calib.py` (motors off): sweep each joint by hand, press Done; writes `~/leg_limits.json` |
| Hang the leg free | stand | carriage on its stop, foot off the floor: the fit assumes a fixed, level base |
| Preview | stand computer | `~/rebot-venv/bin/python leg_record.py --plan` prints the segments from the live pose |
| Record | stand computer | `~/rebot-venv/bin/python leg_record.py --out ~/sysid`, then type `GO` (about 40 s) |
| Fit | laptop | `uv run python -m locomotion.sysid.leg_fit <stem> --output <dir>` |

`leg_record.py` is standalone (numpy and motorbridge); copy it to the stand computer. Every motor runs
RobStride MIT mode with the training motor law: stiffness 12 Nm/rad, the per-joint damping in
`robots/hexapod.py`, zero velocity target, no feedforward. Targets change at 50 Hz and are slew-limited
to 0.040 rad per 20 ms like the policy's; commands are re-sent and feedback read at 100 Hz. One joint
at a time is excited (hold, ±0.15 rad steps, a 0.3–3 Hz chirp of 0.08 rad) while the other two hold.
Amplitudes shrink to stay inside the measured limits less max(7°, 10 %); a joint within 0.03 rad of
that band is skipped. Any of these stops everything: a joint 0.4 rad off target, past its limits by
0.05 rad, above 4.5 Nm or 70 °C, or silent for 0.2 s, or Ctrl+C or SIGTERM. Stopping sets zero
stiffness, then disables every motor. `--fake` runs the whole loop on simulated motors.

`leg_fit.py` replays the recorded commands on a fixed-base MuJoCo model of the lf chain, built by
`leg_sim.py` from the approved model json, through the same motor law at 400 Hz. It fits:

1. **Joint mapping and stiffness from statics.** On settled samples the motor torque balances
   gravity. Least squares gives the femur and tibia sign, motor-frame offset and realised stiffness,
   and the report lists the residual for every sign hypothesis. The tibia carries under 0.05 Nm of
   gravity, so friction can leave its sign ambiguous. The coxa yaw axis is vertical, so gravity
   cannot identify the coxa mapping: verify it by eye.
2. **Delay and dynamics.** Each joint's excitation window is replayed, and Nelder–Mead fits that
   joint's delay, damping scale, armature and Coulomb friction. The three delays are fitted
   independently: their median is the reported delay, and their spread is a check.

`fit.json` keeps:
- the fitted parameters;
- the per-joint position RMS of the nominal training actuator and of the fit;
- the delay in 20 ms controls, for `DynamicsRandomization.action_delay_controls`.

The delay is end to end, as a policy would see it: USB-CAN bridge, bus round trip, motor and feedback
read. `fit.html` overlays target, measurement, nominal simulation and fitted simulation per joint.
`src/locomotion/tests/test_leg_sysid.py` recovers known parameters from a synthetic recording,
including a 23 ms delay to within 4 ms. A fit is evidence about the training actuator model, not a
change to it: `robot/active_model.json` changes only through the program lead.

Check the statics before trusting a fit. The free leg's own gravity torque is at most about 0.3 Nm at
the femur and 0.05 Nm at the tibia. A settled-torque residual above 0.16 Nm sets `loaded_warning`
(free-hanging recordings sit near 0.11 Nm from friction; loaded ones near 0.22 Nm).

### Fixtures and stages

`leg_sim.py` builds one of three fixtures. The stand CAD (`HEXAPOD_MKII_STAND_ASSY`) is not in the
repository, so each one is built from the model json and the 2026-09-04 stand specification:

| Fixture | Mount | Toe | Use |
| --- | --- | --- | --- |
| `clamped` | fixed, level | free in the air | the leg hanging free; actuator and mapping fits |
| `floor` | fixed, level | 12 mm sphere on a floor plane at a fitted depth | the carriage still on its stop, toe pressing the floor |
| `rail` | carriage on a passive vertical slide | on the floor | a carriage carried by the leg (needs its mass) |

`leg_fit.py RECORDING --signs 1,1,-1 [--loaded ...]` fits the free-hanging recording (statics, then the
per-joint delay, damping, armature and friction), and optionally the rail stage on loaded recordings.
`fit_floor` fits one floor depth shared by every foot-down recording, plus toe friction and a small
tibia offset shift. The signs come from the taught jump poses: crouch to extended lowers the femur motor
angle and straightens the tibia while lowering its motor angle.

### Carriage height from the camera

Run Andre's tracker (`testing/leg_stand/tag_height.py serve`, branch `ab3233/leg-stand-tag-height`) with
both tags locked. `leg_record.py` stores `wall_clock_at_t0`. After a recording, save the tracker's
`/history` and run `height_log.py` for a few seconds to map the tracker clock to wall time. Measure the
stand computer's clock offset from the tracker host's, then place each height on the recording's clock.
On macOS the camera needs a terminal app with Camera permission, not a background process.

### Viewing

`stand_view.py RECORDING FIT_JSON --fixture clamped|floor|rail --output DIR` writes `<stem>_real.json`
and `<stem>_sim.json` for the viewer's Runs page (`?run=` both). The lf leg is drawn from the model's
joint frames, with the other legs as mounts. Each case carries its own charts: target, real and
simulated joint angles, plus carriage height on the rail. `leg_video.py` renders the same comparison
as a three-panel video.

### Results, 2026-11-09 (lf leg, CH340 USB-CAN bridge, MIT mode at 100 Hz)

- **Motor law:** torque readings match kp (target − q) − kd dq at stiffness 0.81–1.12 × 12 Nm/rad
  (R² 0.87–0.97) when the target lags 35–45 ms. Position starts following a target step 51–61 ms after
  it. Feedback read right after a command is the previous cycle's reply, up to 10 ms old.
- **Free-hanging fit** (`leg_20261109T143508`, signs +1, +1, −1): per-joint delays of 17, 50 (an
  outlier) and 19 ms, median 19 ms. Position RMS falls from 0.37°/0.97°/0.34° with the training
  actuator to 0.14°/0.72°/0.22°.
- **Held-out check:** `leg_20261109T151125`, also free hanging and camera-confirmed still, falls from
  0.40°/1.02°/0.47° to 0.16°/0.74°/0.46°. The fit holds for the coxa and femur but does not improve
  the tibia.
- **Foot-down recordings** (`…T133352`, `…T143334`, `…T151630`): the camera showed the carriage
  moving under 2 mm while the leg carried up to 0.9 Nm. The carriage rested on its stop and the toe
  pressed on the floor, so the `rail` fixture fits drove the carriage mass to its bound.
  - The `floor` fixture fits one floor depth of 84.1 mm below the mount for all three recordings (start
    toe heights 76–82 mm) and toe friction 0.52. Cost is a quarter of the rail fit's.
  - Position RMS is 0.33–0.43° (coxa), 0.47–0.98° (femur) and 0.76–1.03° (tibia). The fitted tibia is
    worse than the training actuator on two of the three recordings.
- **For training:** the actuator delay is about 1 control; with stale feedback, the command-to-observed
  delay is about 2 controls. Training draws `action_delay_controls` of 0–1.

The fitted values, their recordings and the surrogate training flags that apply them are in
`records/leg_stand_20261109/actuator_params.json`. The trainer can apply the delay (1–2 controls), damping
(0.5–0.7×, its lowest allowed scale; the fit says 0.13–0.37×) and toe friction (0.45–0.6×, which needs
`--contact touchdown_overshoot=False`). It has no setting yet for joint friction or rotor inertia.

Not yet calibrated: a carriage carried by the leg (`rail`), which needs the carriage mass and a
recording with the carriage free, or contact stiffness and foot slip.

References: [detailed wiring/research](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/f5af556/docs/LEG_STAND_HARDWARE.md), [leg test sequence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/project_review_2026-09-04/LEG_TEST_STAND.md), [CAD handoff](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/CAD_ENGINEER_HANDOFF.md).
