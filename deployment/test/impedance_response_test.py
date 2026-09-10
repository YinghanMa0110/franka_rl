"""
Impedance controller response characterisation.

Purpose: diagnose why the real-robot Reach task stalls at ~6cm.
For each stiffness K, command a fixed target offset and measure how far
the end-effector actually travels. This determines:
  (a) the static friction deadband (how far the target must lead before motion)
  (b) the command-to-actual displacement ratio
  (c) whether raising K lets a 1.5cm offset move the arm

offset=0.015 corresponds to the actual per-step offset during deployment
(action ~0.31 x ACTION_SCALE 0.05).
"""

import panda_py
from panda_py import controllers
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# ============ Configuration ============
HOSTNAME = '192.168.1.8'

# Translational stiffness values to test
K_VALUES = [600.0, 900.0, 1200.0, 1800.0]

# Target lead distances to test (metres)
OFFSETS = [0.015, 0.03, 0.05]

# Rotational stiffness
K_ROT = 30.0

CONTROL_FREQ = 20
SETTLE_TIME = 2.0
AXIS = 0               # 0=x, 1=y, 2=z


def run_one(panda, ctrl, pos, orientation, offset):
    """
    Command one fixed Cartesian target.

    Returns:
        moved: actual displacement along test axis
        actual: final Cartesian position
    """
    target = pos.copy()
    target[AXIS] += offset

    with panda.create_context(
        frequency=CONTROL_FREQ,
        max_runtime=SETTLE_TIME
    ) as ctx:

        while ctx.ok():
            ctrl.set_control(target, orientation)

    actual = panda.get_position()

    moved = actual[AXIS] - pos[AXIS]

    return moved, actual


def main():
    axis_name = ['x', 'y', 'z'][AXIS]

    print("=" * 64)
    print("Cartesian impedance response test")
    print(f"  Test axis:    {axis_name}")
    print(f"  Stiffness:    {K_VALUES}")
    print(f"  Offsets (cm): {[o * 100 for o in OFFSETS]}")
    print(f"  Frequency:    {CONTROL_FREQ} Hz")
    print(f"  Settle time:  {SETTLE_TIME} s")
    print("=" * 64)

    print(
        "WARNING: this test raises impedance stiffness. "
        "Stay by the e-stop."
    )

    input(">>> Press Enter to start (Ctrl+C to abort) <<<")

    print("Connecting to robot...")

    panda = panda_py.Panda(HOSTNAME)

    panda.move_to_start()

    print("At start pose")

    ctrl = controllers.CartesianImpedance()

    panda.start_controller(ctrl)

    results = []

    try:
        for k in K_VALUES:

            print(f"\n=== Stiffness K = {k:.0f} ===")

            for offset in OFFSETS:

                # Reset to the same pose before EVERY trial
                panda.stop_controller()

                panda.move_to_start()

                panda.start_controller(ctrl)

                # Set stiffness for this individual trial
                ctrl.set_impedance(
                    np.diag([
                        k,
                        k,
                        k,
                        K_ROT,
                        K_ROT,
                        K_ROT
                    ])
                )

                # Record initial pose
                pos = panda.get_position()
                orientation = panda.get_orientation()

                print(
                    f"\n  K={k:.0f}, "
                    f"lead={offset * 100:.1f}cm"
                )

                print(f"  Start: {pos.round(4)}")

                # Run trial
                moved, actual = run_one(
                    panda,
                    ctrl,
                    pos,
                    orientation,
                    offset
                )

                ratio = moved / offset if offset > 0 else 0.0

                # Target coordinate along selected axis
                target_axis = pos[AXIS] + offset

                # Final remaining target error
                remaining = target_axis - actual[AXIS]

                flag = (
                    "STALLED"
                    if abs(moved) < 0.002
                    else ""
                )

                print(
                    f"  lead {offset * 100:4.1f}cm"
                    f" -> moved {moved * 100:6.2f}cm"
                    f" | remaining {remaining * 100:6.2f}cm"
                    f" | ratio {ratio:5.2f} "
                    f"{flag}"
                )

                results.append(
                    (
                        k,
                        offset,
                        moved,
                        ratio
                    )
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\n\nError: {e}")

    finally:
        print("\n" + "=" * 64)

        # Stop impedance controller
        try:
            panda.stop_controller()
        except Exception:
            pass

        print("Returning to start...")

        try:
            panda.move_to_start()

        except Exception:
            print(
                "move_to_start failed, attempting recover..."
            )

            try:
                panda.recover()
                panda.move_to_start()

            except Exception as e:
                print(
                    f"recover also failed: {e}"
                )

        # Save results
        if results:
            arr = np.array(results)

            np.save(
                'impedance_response.npy',
                arr
            )

            print(
                "Raw data saved to impedance_response.npy"
            )

            # Summary
            print(
                "\nSummary "
                "(ratio = actual movement / commanded lead)"
            )

            header = (
                "  K      "
                + "".join(
                    f"{o * 100:>10.1f}cm"
                    for o in OFFSETS
                )
            )

            print(header)
            print(
                "  "
                + "-" * (len(header) - 2)
            )

            for k in K_VALUES:
                row = arr[
                    arr[:, 0] == k
                ]

                cells = "".join(
                    f"{r[3]:>12.2f}"
                    for r in row
                )

                print(
                    f"  {k:<6.0f}{cells}"
                )

            print("\nHow to read this table:")

            print(
                "  - Look at the 1.5cm column: "
                "that is the actual deployment offset"
            )

            print(
                "    ratio clearly > 0 at some K "
                "-> stiffness may be important"
            )

            print(
                "    ratio ~0 at every K "
                "-> investigate target accumulation + MAX_LEAD"
            )

            print(
                "  - Ratio stable across offsets "
                "-> approximately linear response"
            )

            print(
                "  - Small offsets ~0, larger offsets non-zero "
                "-> possible effective deadband"
            )

        print("Done")


if __name__ == '__main__':
    main()