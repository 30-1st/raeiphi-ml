# RAEIPHI — SaaS Platform Features

## Dashboard Overview

The RAEIPHI dashboard is the central interface for monitoring your properties. It provides real-time occupancy data, historical analytics, alert management, and access to the RAEIPHI AI assistant. The dashboard is accessible from any web browser on desktop or mobile devices.

## Real-Time Occupancy Monitoring

The primary dashboard view shows the current occupancy count for each property and each room with nodes installed. The count updates in real time as people enter and leave. Each property card displays the current headcount, the last updated timestamp, and a color-coded status indicator: green for within expected occupancy, yellow for approaching the threshold, and red for exceeding the set limit.

Clicking into a property shows a room-by-room breakdown. If you have labeled your nodes by location during setup, you can see which rooms are occupied and approximately how many people are in each area.

## Alerts and Notifications

RAEIPHI supports configurable alerts that notify you when occupancy conditions change. Alerts can be delivered via email, push notification, or both.

**Occupancy threshold alert:** Set a maximum guest count for your property. When the detected occupancy exceeds this number, you receive an immediate notification. This is the primary alert for detecting unauthorized guests or parties.

**Unexpected presence alert:** If your property should be vacant (between bookings, during a maintenance window), any detected occupancy triggers an alert. This is useful for detecting trespassers or guests who have not checked out.

**Check-in and check-out verification:** Set expected check-in and check-out times. RAEIPHI notifies you when guests arrive and when the property is vacated, helping you coordinate cleaning crews and turnover logistics.

**Anomaly detection:** The system uses a machine learning model trained to recognize unusual occupancy patterns. A sudden spike from 2 to 12 people at midnight triggers an anomaly alert even if you have not set a specific threshold. This catches events you might not have anticipated.

## Historical Data and Analytics

RAEIPHI stores occupancy data over time, allowing you to review patterns and trends. The historical view shows occupancy over hours, days, weeks, or months.

**Occupancy timeline:** A chart showing how many people were detected at each time interval. Useful for verifying guest behavior, confirming check-in/check-out times, and identifying patterns like recurring unauthorized visitors.

**Peak occupancy reports:** Summary statistics showing the maximum occupancy detected during each booking, average occupancy across bookings, and time-of-day patterns.

**Booking correlation:** If you connect your booking calendar, RAEIPHI can overlay occupancy data against booked guest counts. Discrepancies between booked guests and detected occupancy are flagged automatically.

## Multi-Property Management

The RAEIPHI subscription covers all properties under a single account at no additional per-property fee. Property managers can view all their units from one dashboard, with summary cards showing the status of each property at a glance.

Properties can be grouped by location, building, or any custom categorization. Alerts are configurable per property — you can set different thresholds for a studio versus a large vacation home.

## RAEIPHI AI Assistant

The built-in AI assistant helps with property management questions, system configuration, occupancy interpretation, and product support. The assistant has access to your property data and can answer questions like:

- "How many people were in my downtown unit last Saturday night?"
- "Set an alert if occupancy exceeds 6 people in any of my properties."
- "What is the average occupancy during weekday bookings?"
- "Why did I get an anomaly alert at 2 AM?"

The AI assistant also supports prospective customers considering RAEIPHI. It can explain the technology, recommend node configurations for specific property layouts, walk through pricing options, and assist with purchasing decisions.

For custom or large-scale deployments, the AI assistant can discuss pricing, recommend configurations, and help scope a deployment plan tailored to specific needs.

## Node Health Monitoring

The dashboard shows the status of every node in every property. Each node reports its connectivity status, signal strength, and last communication time. If a node goes offline, loses WiFi, or experiences hardware issues, the dashboard flags it and sends a notification.

Node health monitoring ensures you are always aware of your sensing coverage. A property with an offline node may have gaps in occupancy detection — the system warns you so you can address it before it becomes a blind spot.

## Data Security

All data transmitted between RAEIPHI nodes and the cloud platform is encrypted in transit. Occupancy data stored in the platform is encrypted at rest. The system does not collect, store, or transmit any personally identifiable information about occupants. The only data captured is signal measurements and the resulting occupancy count.

Account access is protected by authentication. Property data is isolated between accounts — no user can access another user's property data.
