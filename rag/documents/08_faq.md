# RAEIPHI — Frequently Asked Questions

## General

### What is RAEIPHI?
RAEIPHI is a WiFi-based occupancy sensing system for short-term rentals, property managers, hotels, and homeowners. It uses WiFi Channel State Information (CSI) to detect how many people are in a property in real time — without cameras, microphones, or any recording devices.

### How does RAEIPHI detect people without cameras?
RAEIPHI works by measuring how WiFi signals travel between its hardware nodes. Human bodies reflect, absorb, and scatter WiFi radio waves in measurable ways. A machine learning model analyzes these signal changes across 52 WiFi frequency subcarriers to determine how many people are present. No images, audio, or video are captured at any point.

### Does RAEIPHI record anything?
No. RAEIPHI captures WiFi signal measurements only — amplitude, phase, and timing data from the radio signals traveling between nodes. This raw signal data is processed into an occupancy count. The signal data itself does not contain any visual, audio, or personally identifiable information.

### Can RAEIPHI identify who is in the room?
No. The system determines how many people are present, not who they are. It cannot distinguish between individuals, detect personal devices, or capture any biometric data. Two different groups of the same size would produce equivalent readings.

### Is RAEIPHI compliant with Airbnb's monitoring policies?
Yes. Airbnb prohibits surveillance devices, defined as devices that capture images, video, or audio, inside rental properties. RAEIPHI is not a surveillance device — it captures WiFi signal measurements, not sensory recordings. It provides occupancy data, not identification or behavioral monitoring.

## Pricing and Purchasing

### How much does RAEIPHI cost?
The hardware is a one-time purchase. Individual DIY nodes are $49 each plus shipping and handling. Full packages with setup instructions and a SaaS account login are $89 each plus shipping and handling. The SaaS platform subscription is $19.99 per month, which covers all nodes and all properties under your account.

### How many nodes do I need?
A minimum of 4 nodes per property is required for reliable occupancy detection. The exact number depends on your property size and layout. A studio or small one-bedroom typically needs 4 nodes. A two-bedroom needs 4 to 5. A three-bedroom house needs 5 to 6. Larger properties may need 6 to 8 or more.

### Is there a monthly fee per node or per property?
No. The $19.99 monthly subscription covers your entire account — all nodes and all properties. Whether you have 4 nodes in one apartment or 30 nodes across five properties, the monthly cost is the same.

### Can I start with the minimum and add nodes later?
Yes. You can purchase additional DIY nodes at any time and add them to your existing property setup. The system automatically incorporates new nodes after a brief re-calibration.

### Do you offer volume or custom pricing?
Yes. For property management companies, hotel operators, multi-unit buildings, or any deployment requiring more than 8 nodes, RAEIPHI offers custom pricing. The RAEIPHI AI assistant can discuss your specific needs and provide a custom quote.

### What is your refund policy?
Hardware may be returned within 30 days of delivery for a full refund if in original condition. The SaaS subscription can be cancelled at any time with no cancellation fee or long-term contract.

## Setup and Installation

### Is RAEIPHI hard to install?
No. Installation consists of plugging nodes into power outlets and connecting them to your WiFi network through a guided setup process. No wiring, drilling, or technical expertise is required. The full package includes printed instructions, and the RAEIPHI AI assistant can walk you through setup step by step.

### Does RAEIPHI need its own WiFi network?
No. RAEIPHI nodes connect to your property's existing 2.4 GHz WiFi network. No additional router, hub, or networking equipment is required.

### Will RAEIPHI slow down my WiFi?
No. The signal measurements RAEIPHI uses are a byproduct of normal WiFi communication. The nodes send minimal data to the cloud — just processed occupancy readings, not raw signal data. Bandwidth usage is negligible.

### What happens if my WiFi goes down?
If the WiFi network goes offline, the nodes cannot transmit occupancy data to the cloud platform. The dashboard will show the nodes as offline. Occupancy monitoring resumes automatically when WiFi connectivity is restored. No data is lost — the system simply has a gap during the outage.

### Can I move nodes after installation?
Yes. If you rearrange furniture or want to optimize placement, you can unplug a node and move it to a different outlet. After moving nodes, run a re-calibration from the dashboard to establish new baseline signal measurements.

## Accuracy and Performance

### How accurate is RAEIPHI?
RAEIPHI's machine learning model achieves over 97% classification accuracy across occupancy levels 0 through 10 on test data. Accuracy is highest for low occupancy levels and slightly lower for high counts (7+), but for the primary use case of detecting whether a property has significantly more people than expected, the accuracy is robust.

### Does RAEIPHI detect pets?
Large pets (roughly 30 pounds or more) may occasionally be detected as fractional occupancy. Small pets such as cats and small dogs generally do not affect readings. If you have a large dog in your property, the system may show an occupancy of 1 when only the dog is present.

### Can RAEIPHI work through walls?
Yes. WiFi signals pass through standard building materials including drywall, wood, glass, and most interior walls. A node in the living room can contribute to sensing in an adjacent room. However, thick concrete, metal-reinforced walls, and large metal objects significantly attenuate the signal. Each room where you want reliable detection should have at least one node or be within clear signal range of multiple nodes.

### What is the maximum occupancy RAEIPHI can detect?
The current model is trained for occupancy levels 0 through 10. Above 10 people, the system reports 10+ but does not distinguish between specific counts above that threshold. For most short-term rental use cases, knowing that a property has 10+ people is sufficient to trigger an alert.

### Does furniture or room layout affect accuracy?
Normal furnished rooms do not significantly impact accuracy. The calibration step accounts for the static environment, including furniture. However, very large metal objects (metal shelving units, large appliances directly between nodes) can attenuate signals. Nodes should be placed to avoid direct obstruction by large metal objects.

## Security and Privacy

### What data does RAEIPHI collect?
RAEIPHI collects WiFi signal measurements between its nodes and the resulting occupancy count. It does not collect images, video, audio, or any personally identifiable information about occupants.

### Where is my data stored?
Occupancy data is stored in RAEIPHI's cloud platform, which uses encrypted storage. Data is accessible only through your authenticated account.

### Can guests tamper with RAEIPHI nodes?
The nodes are designed to be discreet — small devices in neutral-colored casings that blend with standard home electronics. However, like any physical device, a determined person could unplug a node. If a node goes offline, the dashboard and alert system notify you immediately.

### Do I need to disclose RAEIPHI to guests?
Disclosure requirements vary by jurisdiction and platform. Since RAEIPHI does not capture images, video, or audio, it is generally not classified as a surveillance device. However, some jurisdictions require disclosure of any monitoring system. Check your local regulations and platform policies. When in doubt, a brief mention that the property uses a WiFi-based occupancy monitoring system for safety and compliance is typically sufficient and builds guest trust.

## Technical

### What WiFi standard does RAEIPHI use?
RAEIPHI currently uses 2.4 GHz WiFi (802.11n). The nodes must connect to a 2.4 GHz network. Most modern routers broadcast both 2.4 GHz and 5 GHz — ensure the 2.4 GHz band is enabled.

### What hardware is inside a RAEIPHI node?
Each node contains an ESP32 microcontroller and a Raspberry Pi, housed in a custom RAEIPHI-branded casing. The ESP32 handles WiFi CSI measurement, and the Raspberry Pi handles local signal processing and cloud communication.

### Does RAEIPHI work with mesh WiFi systems?
Yes, as long as the mesh system supports 2.4 GHz. RAEIPHI nodes need to connect to the same network, but they do not require line-of-sight to the router — they measure signals between each other, not between the nodes and the router.

### Can I integrate RAEIPHI with my smart home system?
Integration capabilities are on the RAEIPHI roadmap. Future updates will support standard smart home protocols for triggering automations based on occupancy data.
