
## **Distance Thresholds**

### Cable Connectivity (Step 1)
- **0.5m** - Snap tolerance for connecting cable endpoints to form 
- **0.005m** - Snap tolerance for connecting cable endpoints to form networks
- **0.01m** - Minimum cable segment length to include

### Building Connections (Steps 4-8)
- **10 m²** - Minimum building area to process
- **150m** - Maximum "normal" connection distance (beyond this is flagged as "TOO_FAR")
- **500m** - Search radius for finding LV connections
- **1000m** - Absolute maximum connection distance allowed
- **100m** - Search radius for MV connections (for MV-capable buildings)

### Station Connections (Steps 2-3)
**LV to LV Cabinets:**
- 1m - Direct connection
- 10m - High confidence proximity
- 25m - Medium confidence
- 50m - Low confidence
- 100m - Maximum search radius

**LV to Transformers:**
- 2m - Direct connection
- 20m - Very high confidence
- 50m - High confidence
- 100m - Medium confidence
- 200m - Low confidence
- 300m - Maximum search radius

**MV to Transformers:**
- 2m - Direct connection
- 10m - Very high confidence
- 30m - High confidence
- 100m - Medium confidence
- 200m - Maximum search radius

**MV to Substations:**
- 5m - Direct connection
- 50m - Very high confidence
- 200m - High confidence
- 500m - Medium confidence
- 1000m - Maximum search radius

**HV to Substations:**
- 10m - Direct connection
- 100m - High confidence
- 500m - Medium confidence
- 2000m - Maximum search radius

## **Confidence Scores**

The confidence scores decrease with distance:
- **0.95-0.99** - Direct/very close connections
- **0.85-0.90** - Close proximity
- **0.65-0.75** - Medium distance
- **0.40-0.60** - Far distance
- **0.20-0.30** - Very far/edge cases

## **Building Classification**

### MV Capability Threshold
- **3000 m²** - Non-residential buildings larger than this are flagged as MV-capable

## **Connection Type Classifications**

Buildings are classified by how they connect:
- **ENDED** - Cable endpoint within building footprint
- **ENTERED** - Cable enters building
- **CROSSED** - Cable crosses through building
- **BY_DISTANCE** - Connected by proximity (≤150m)
- **TOO_FAR** - Distance >150m (problematic)

## **Search Limits**

- **3** - Maximum LV cabinets to connect per LV group
- **5** - Maximum transformers to connect per MV group
- **2** - Maximum substations to connect per HV group
- **1** - Closest connection selected for most building connections

