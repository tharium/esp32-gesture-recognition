# ESP32 Gesture Recognition

A deep learning project for WiFi-based gesture recognition using ESP32 microcontrollers and CSI (Channel State Information).

---

## Setup

This project requires **two ESP32 boards**:

- One configured as an **Access Point (AP)**
- The other as a **Station (STA)** to receive CSI data

CSI collection is based on a modified version of [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) by Steven M. Hernandez.

A modified version of the `csi_component.h` file is included in the `misc/` directory. See the [Credits](#credits) section for license and attribution information.

---

## Dataset

The repository includes **5 CSV files**, each containing ~100 samples of CSI data for the following gestures:

- **Control** (no gesture)
- **Wave**
- **Clap**
- **Push** (moving hand toward AP)
- **Pull** (moving hand away from AP, toward STA)

**Collection Details**:
- Boards were placed ~1.5 feet apart on a level surface.
- Each sample contains ~100 packets of CSI data.

**Limitations**:
- Dataset size is small (~50,000 CSI packets total).
- Performance is **environment-sensitive**. Different physical settings may significantly affect prediction accuracy.
- The included test script achieves ~**75% accuracy** using this dataset — suitable for testing and demonstration, but not production.

---

## Usage

1. Flash ESP32 boards using the ActiveAP and ActiveSTA from the ESP32-CSI-Tool repo, optionally add modified CSI firmware (`csi_component.h`).
2. Run the data collection script to gather CSI samples (`log_serial.py`).
3. Run the clean script to remove unecessary data from the csv (`clean_csv.py`).
3. Use the included dataset or record your own.
4. Train and test the model (`test_train.py`).
5. Note that `test_train.py` runs a Random Forest Baseline test before running the NN, if you comment/remove that portion be sure to set the threshold for the NN to run at 0.

*More detailed instructions will be added, eventually*

---

## Credits

This project includes code modified from the [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) by **Steven M. Hernandez**, licensed under the MIT License.