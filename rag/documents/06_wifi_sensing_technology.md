# RAEIPHI — WiFi Sensing Technology

## What is WiFi Channel State Information (CSI)?

Every WiFi transmission between two devices carries more than just data. The radio signal traveling from a transmitter to a receiver is affected by everything in the environment — walls, furniture, and critically, people. Channel State Information (CSI) is the detailed measurement of how the WiFi signal was altered during transmission.

In technical terms, CSI captures the amplitude and phase of the signal across multiple frequency subcarriers. A standard 20 MHz WiFi channel (802.11n) uses 52 subcarriers — 52 individual frequency slices that each carry a portion of the signal. Each subcarrier responds slightly differently to environmental conditions, and CSI records these responses as complex numbers representing amplitude (signal strength) and phase (signal timing).

When nothing is moving in the environment, CSI readings are stable. When a person walks through the signal path, the readings change. More people create more changes. RAEIPHI's machine learning model is trained to interpret these changes and determine how many people are present.

## How CSI Detects People

Human bodies interact with WiFi signals in three measurable ways:

**Reflection:** Radio waves bounce off the human body, creating additional signal paths (multipath). Each person in the room adds their own set of reflections, increasing the complexity of the signal pattern. This is measurable as increased variance in signal amplitude across subcarriers.

**Absorption:** The human body (being mostly water) absorbs some of the WiFi signal energy. More people in the room means more absorption, which shows up as a slight decrease in average signal amplitude.

**Scattering:** When people move, they scatter the signal in changing directions. This creates temporal variation — the CSI readings fluctuate more when people are moving versus when the room is still. The rate and magnitude of fluctuation correlates with the number and activity level of the occupants.

By measuring all three effects simultaneously across 52 subcarriers and multiple node pairs, RAEIPHI builds a rich picture of the environment. The machine learning model has been trained on thousands of examples to distinguish between 0 people, 1 person, 2 people, and so on up to 10 or more.

## Why WiFi Sensing Works Better Than Alternatives

**Versus cameras:** WiFi sensing captures no visual information. There is nothing to see, record, or store. It cannot identify who is in the room, what they are wearing, or what they are doing — only how many people are present. This makes it fundamentally different from surveillance. Cameras are also limited to line-of-sight and cannot see through walls. WiFi signals pass through walls, doors, and most building materials, providing sensing coverage even in rooms without direct line-of-sight to a node.

**Versus motion sensors (PIR):** Passive infrared sensors detect the presence of a warm body moving in their field of view. They provide a binary reading — motion detected or not. They cannot count people. A room with 1 person and a room with 8 people look the same to a motion sensor. They also require line-of-sight and have limited range. WiFi sensing provides an actual occupancy count across the entire property.

**Versus noise monitors:** Noise-based systems detect disturbances after they escalate. By the time a party is loud enough to trigger a noise alert, the event is already underway. WiFi sensing detects the occupancy increase that precedes the noise, providing earlier warning.

**Versus Bluetooth or phone-based tracking:** These systems count devices, not people. They miss anyone without a Bluetooth-enabled device or anyone who has Bluetooth turned off. They also raise privacy concerns because they interact with personal devices. WiFi sensing is entirely passive and does not communicate with or detect any personal device.

**Versus pressure mats or floor sensors:** Physical sensors require installation under flooring, are expensive to retrofit, and can be circumvented. WiFi sensing uses existing WiFi infrastructure and requires only plugging in nodes.

## The Role of Machine Learning

Raw CSI data is noisy and complex. A single measurement between two nodes produces 52 amplitude values and 52 phase values across the subcarriers. With 4 nodes creating 6 unique measurement pairs, each time window produces thousands of data points.

A machine learning model is essential because no simple threshold or rule can reliably interpret this data. The model learns patterns that humans cannot manually define — subtle relationships between subcarrier groups, temporal dynamics that distinguish 3 people sitting still from 2 people moving, and environmental characteristics that vary between properties.

RAEIPHI uses a deep learning model (a 1D Convolutional Neural Network) trained on WiFi CSI signal data. The model takes engineered features extracted from raw CSI measurements — signal-to-noise ratios, amplitude variance, phase stability, temporal dynamics, and cross-subcarrier correlations — and outputs a predicted occupancy count with a confidence score.

The model was trained and evaluated using rigorous machine learning methodology: train/validation/test splits with stratification, StandardScaler normalization fitted on training data only, and comparison against classical machine learning baselines (Random Forest, Gradient Boosting). The deep learning model outperforms the baselines with over 97% accuracy on held-out test data.

## Signal Physics — Going Deeper

For those interested in the physics, here is how CSI sensing works at the signal level.

WiFi uses Orthogonal Frequency-Division Multiplexing (OFDM), which divides the radio channel into many narrow subcarriers. In a 20 MHz channel, there are 52 data subcarriers. Each subcarrier can be thought of as an independent mini-signal that travels from transmitter to receiver.

As the signal travels, it takes multiple paths — directly, bouncing off walls, bouncing off furniture, bouncing off people. Each path has a different length, which means each path arrives at a slightly different time and phase. The receiver combines all these paths, and the result is captured in the CSI matrix.

When a person enters the room, they add new reflection paths and block or absorb existing ones. This changes the CSI matrix in a way that is characteristic of human presence. The change is not random noise — it has structure that corresponds to the physical properties of the human body (size, water content, movement patterns).

Multiple people add multiple independent sets of reflections and absorptions. The CSI matrix becomes more complex in measurable ways: higher amplitude variance across subcarriers, greater phase instability, reduced signal-to-noise ratio, and increased temporal variation.

The machine learning model learns to map this complexity to a count. It does not need to "see" each person individually — it recognizes aggregate signal patterns that correspond to specific occupancy levels.

## Accuracy and Limitations

RAEIPHI's occupancy detection is highly accurate under normal conditions — above 97% classification accuracy across occupancy levels 0 through 10.

Accuracy is highest for distinguishing between low occupancy levels (0, 1, 2, 3) and decreases slightly for higher counts (7, 8, 9, 10). This is because the incremental signal change from person 9 to person 10 is smaller than from person 0 to person 1. For most use cases — detecting whether a property has significantly more people than booked — the accuracy is more than sufficient.

Factors that can affect accuracy:
- **Node count and placement:** Fewer than 4 nodes or poor placement significantly reduces accuracy. Follow the placement guidelines.
- **Wall materials:** Thick concrete, metal-reinforced walls, and large metal objects can attenuate WiFi signals and reduce sensing coverage. Standard drywall, wood, and glass are not problematic.
- **WiFi interference:** Very congested WiFi environments with many competing networks can introduce noise. RAEIPHI's signal processing pipeline includes noise reduction, but extreme interference can affect performance.
- **Pets:** Large pets (over approximately 30 pounds) may occasionally register as fractional occupancy. Small pets generally do not affect readings.
- **Large gatherings:** The model is trained for occupancy up to 10. Above 10 people, it reports 10+ but does not attempt to distinguish between 12 and 15.
