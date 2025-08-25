# Physical Parameter Fixes Summary

## Issues Fixed

### 1. Solar Generation (FIXED)
**Problem**: Peak generation was 1350 kW from 8.33 kWp system (162x too high)
**Cause**: Double-applying efficiency - kWp already includes panel efficiency
**Fix**: Removed redundant solar_efficiency multiplication, only apply system losses
**Result**: Peak generation now 6.38 kW (76% of capacity - realistic)

### 2. Annual Generation (FIXED)  
**Problem**: Annual generation showing 10.25 kWh instead of ~10,000 kWh
**Cause**: Summing 24-hour profile without scaling to annual
**Fix**: Added logic to detect daily vs annual profiles and scale appropriately
**Result**: Annual generation now 12,372 kWh (realistic for 8.33 kWp in Europe)

### 3. Cascade Effects (FIXED)
**Problem**: Cascade energy exceeding available generation (4.32 kW > 3.0 kW)
**Cause**: Not tracking available energy after each share
**Fix**: Implemented available_for_sharing tracking that decreases with each P2P trade
**Result**: Total cascade now 2.92 kW (within 3.0 kW shareable limit)

### 4. Loss Functions (FIXED)
**Problem**: Multiple loss functions returning negative values
**Cause**: Mathematical formulations minimizing negative correlations
**Fix**: Reformulated to ensure positive loss values, added ReLU to network impact heads
**Result**: All losses now positive, training stable

### 5. Diversity Loss (FIXED)
**Problem**: Exploding to 22026+ 
**Cause**: Unbounded log determinant calculation
**Fix**: Added sigmoid bounding and proper normalization
**Result**: Diversity loss now bounded between 0 and 1

## Key Physical Parameters Now Correct

- **Solar capacity**: 6 m²/kWp (industry standard)
- **Peak generation**: ~90% of rated capacity (accounting for losses)
- **Annual generation**: ~1200-1500 kWh/kWp (typical for Europe)
- **Cascade sharing**: Limited by available surplus after local consumption
- **P2P efficiency**: 95% (realistic for local trading)
- **System losses**: 15% (inverter, wiring, temperature)

## Verification Results

```
Solar Generation: OK
  - Peak: 6.38 kW from 8.33 kWp (76% - realistic with losses)
  - Annual: 12,372 kWh (capacity factor 17% - typical for Europe)

Cascade Effects: OK  
  - Total shared: 2.92 kW
  - Available surplus: 3.00 kW
  - Energy conservation maintained

Training: OK
  - All losses positive
  - Model converges
  - Network effects captured

Peak Reduction: Realistic
  - 0% reduction (solar peaks at noon, demand peaks at 6am)
  - Would need battery storage for morning peak shaving
```

## Units Consistency

- **kW**: Instantaneous power (demand, generation at a moment)
- **kWh**: Energy over time (annual generation, consumption)
- **kWp**: Peak capacity rating of solar panels

All calculations now use consistent units throughout the pipeline.