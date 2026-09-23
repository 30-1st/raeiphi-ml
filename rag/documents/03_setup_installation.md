# RAEIPHI — Node Setup and Installation Guide

## What Is In the Box

Each RAEIPHI node package includes:
- One RAEIPHI sensing node (ESP32 + Raspberry Pi in a custom RAEIPHI-branded casing)
- One power adapter with cable
- Quick start card with QR code linking to setup instructions
- Full package orders also include a printed placement guide and account login credentials

## Hardware Requirements

Before installing RAEIPHI nodes, ensure your property meets these requirements:

**WiFi network:** A standard 2.4 GHz WiFi network is required. The nodes connect to your existing WiFi — no additional router or network equipment is needed. The network must have internet access for the nodes to communicate with the RAEIPHI cloud platform.

**Power outlets:** Each node needs a standard wall outlet. The nodes draw minimal power (comparable to a phone charger) and are designed to run continuously.

**Minimum 4 nodes:** For viable occupancy detection accuracy, you need at least 4 nodes per property. Fewer than 4 nodes will not provide reliable occupancy counts because the machine learning model requires signal measurements from multiple angles and positions to triangulate occupancy.

## Node Placement Guidelines

Proper node placement is critical for accurate occupancy sensing. The nodes work by measuring how WiFi signals travel between them — each pair of nodes creates a "sensing link" that detects disturbances caused by human presence. More links with better spatial diversity means higher accuracy.

**General principles:**

Spread nodes across the property rather than clustering them in one room. The goal is to create a mesh of sensing links that covers the entire living space.

Place nodes at varied heights when possible. A node on a shelf at chest height and another on a low table creates vertical signal diversity, which improves detection of seated versus standing occupants.

Avoid placing nodes directly behind large metal objects, inside closed cabinets, or in corners completely blocked by furniture. The WiFi signal needs a relatively clear path between nodes.

Position at least one node in every room where occupancy detection matters. If a bedroom door is typically closed, that room needs its own node to maintain a sensing link.

**Recommended placements by property type:**

For a studio or one-bedroom (4 nodes):
- Node 1: Living area, near the entrance
- Node 2: Living area, opposite side or near the kitchen
- Node 3: Bedroom, near the door
- Node 4: Bedroom, opposite side or near the window

For a two-bedroom (5 nodes):
- Nodes 1–2: Living area, spread across the main common space
- Node 3: Bedroom 1
- Node 4: Bedroom 2
- Node 5: Hallway or kitchen area connecting the rooms

For a three-bedroom house (6 nodes):
- Nodes 1–2: Main living area and kitchen
- Nodes 3–5: One per bedroom
- Node 6: Hallway, landing, or secondary common area

## Step-by-Step Setup

**Step 1: Create your RAEIPHI account**

Visit the RAEIPHI platform and create your account. If you purchased the full package, your login credentials are included in the box. If you purchased DIY nodes, create an account at the RAEIPHI website.

**Step 2: Add your property**

In the RAEIPHI dashboard, click "Add Property" and enter your property details: name, address, number of rooms, and the number of nodes you are installing. This creates the property profile that your nodes will report to.

**Step 3: Power on each node**

Plug each node into a power outlet in its designated location. The node will power on automatically — a small LED indicator will blink to show it is starting up.

**Step 4: Connect nodes to WiFi**

Each node broadcasts a temporary setup network when first powered on. Connect to this network from your phone or laptop, enter your property WiFi credentials, and the node will join your network. The RAEIPHI app walks you through this process step by step.

**Step 5: Register nodes to your property**

Once connected to WiFi, each node appears in your RAEIPHI dashboard under the property you created. Confirm each node and optionally label its location (e.g., "Living Room - North Wall", "Bedroom 2").

**Step 6: Calibration**

After all nodes are installed and connected, the system runs an automatic calibration period. During calibration (approximately 15–30 minutes), the nodes establish baseline signal measurements for the empty property. For best results, ensure the property is unoccupied during calibration.

**Step 7: Verify**

Once calibration is complete, walk through the property to verify the system detects your presence. The dashboard should show an occupancy count of 1. Have another person enter — the count should update. If any rooms are not detecting properly, adjust node placement and re-calibrate.

## Troubleshooting

**Node not connecting to WiFi:** Ensure your WiFi network is 2.4 GHz. Many modern routers broadcast both 2.4 GHz and 5 GHz — the node requires 2.4 GHz. Check that your WiFi password is entered correctly.

**Inaccurate occupancy count:** The most common cause is insufficient nodes or poor placement. Ensure you have at least 4 nodes and that they are spread across the property rather than clustered. Re-run calibration after adjusting placement.

**Node appears offline in dashboard:** Check that the node is still powered on and that your WiFi network is operational. If the node's LED is solid (not blinking), it is connected but may have lost internet access. Restart your router.

**Occupancy count does not change when property is empty:** Pets above approximately 30 pounds may be detected. Large moving objects (robotic vacuums, ceiling fans creating airflow) can occasionally cause minor readings. The system is tuned to minimize these false positives but they can occur in some environments.
