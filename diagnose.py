#!/usr/bin/env python3
"""
Diagnostic script to check if models learned different behaviors
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.velocity_field import VelocityFieldNet
from models.shortcut_predictor import ShortcutPredictor

def main():
    print("="*80)
    print("DIAGNOSTIC: Testing Model Predictions")
    print("="*80)
    
    # Load Sequential model
    print("\n📥 Loading models...")
    try:
        seq_model = VelocityFieldNet(state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64,64,64,64])
        seq_checkpoint = torch.load('experiments/sequential_baseline_model.pt', map_location='cpu')
        seq_model.load_state_dict(seq_checkpoint['model_state_dict'])
        seq_model.eval()
        print("  ✓ Sequential model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load Sequential model: {e}")
        return
    
    # Load Shortcut model
    try:
        shortcut_velocity_net = VelocityFieldNet(state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64,64,64,64])
        shortcut_model = ShortcutPredictor(shortcut_velocity_net)
        shortcut_checkpoint = torch.load('experiments/shortcut_bootstrap_model.pt', map_location='cpu')
        shortcut_model.load_state_dict(shortcut_checkpoint['model_state_dict'])
        shortcut_model.eval()
        print("  ✓ Shortcut model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load Shortcut model: {e}")
        return
    
    print("\n" + "="*80)
    print("TEST CASE: Particle approaching wall")
    print("="*80)
    print("Initial state: x=4.0, y=0.0, vx=2.0, vy=0.0")
    print("Scenario: Particle moving RIGHT at x=4, will hit wall at x=5")
    print("No external forces applied")
    print("="*80)
    
    # Test case: particle moving right, about to hit wall at x=5
    state = torch.FloatTensor([[4.0, 0.0, 2.0, 0.0]])  # x=4, y=0, vx=2, vy=0
    actions = torch.zeros(1, 20, 2)  # no forces
    time = torch.zeros(1, 1)
    
    # Test at different horizons
    horizons = [0.01, 0.1, 0.5, 1.0]
    
    for horizon in horizons:
        dt_tensor = torch.FloatTensor([[horizon]])
        
        # Sequential prediction
        with torch.no_grad():
            seq_velocity = seq_model(state, actions, time, dt_tensor)
            seq_pred = state + seq_velocity * dt_tensor
        
        # Shortcut prediction  
        with torch.no_grad():
            sc_velocity = shortcut_model.velocity_net(state, actions, time, dt_tensor)
            sc_pred = state + sc_velocity * dt_tensor
        
        print(f"\n{'─'*80}")
        print(f"HORIZON: dt={horizon:.2f}s")
        print(f"{'─'*80}")
        
        print(f"\n🔵 SEQUENTIAL MODEL:")
        print(f"   Velocity vector: [{seq_velocity[0,0]:.4f}, {seq_velocity[0,1]:.4f}, {seq_velocity[0,2]:.4f}, {seq_velocity[0,3]:.4f}]")
        print(f"   Predicted state: x={seq_pred[0,0]:.3f}, y={seq_pred[0,1]:.3f}, vx={seq_pred[0,2]:.3f}, vy={seq_pred[0,3]:.3f}")
        
        print(f"\n🟢 SHORTCUT MODEL:")
        print(f"   Velocity vector: [{sc_velocity[0,0]:.4f}, {sc_velocity[0,1]:.4f}, {sc_velocity[0,2]:.4f}, {sc_velocity[0,3]:.4f}]")
        print(f"   Predicted state: x={sc_pred[0,0]:.3f}, y={sc_pred[0,1]:.3f}, vx={sc_pred[0,2]:.3f}, vy={sc_pred[0,3]:.3f}")
        
        # Check for differentiation
        pos_diff = abs(seq_pred[0,0].item() - sc_pred[0,0].item())
        vel_diff = abs(seq_pred[0,2].item() - sc_pred[0,2].item())
        
        print(f"\n📊 COMPARISON:")
        print(f"   Position difference: {pos_diff:.4f}")
        print(f"   Velocity difference: {vel_diff:.4f}")
        
        # Expected behavior analysis
        if horizon >= 0.5:
            print(f"\n💡 EXPECTED BEHAVIOR at dt={horizon}:")
            print(f"   Wall is at x=5.0")
            print(f"   Particle starts at x=4.0 with vx=2.0")
            print(f"   Without collision: x would be {4.0 + 2.0*horizon:.1f}")
            print(f"   ")
            print(f"   SEQUENTIAL (trained only on dt=0.01):")
            print(f"     Should EXTRAPOLATE → go PAST wall (x>5.0)")
            print(f"     Should NOT predict collision bounce")
            print(f"     ")
            print(f"   SHORTCUT (trained on dt={horizon}):")
            print(f"     Should PREDICT collision → stay NEAR wall (x≈5.0)")
            print(f"     Should predict velocity REVERSAL (vx<0)")
            
            # Analyze actual predictions
            print(f"\n🔍 ACTUAL ANALYSIS:")
            if seq_pred[0,0] > 5.5:
                print(f"   ✓ Sequential went past wall (x={seq_pred[0,0]:.2f}) - CORRECT FAILURE")
            else:
                print(f"   ✗ Sequential didn't extrapolate past wall (x={seq_pred[0,0]:.2f}) - UNEXPECTED")
            
            if abs(sc_pred[0,0].item() - 5.0) < 0.5 and sc_pred[0,2] < 0:
                print(f"   ✓ Shortcut predicted collision (x≈5.0, vx<0) - CORRECT SUCCESS")
            elif sc_pred[0,0] > 5.5:
                print(f"   ✗ Shortcut went past wall (x={sc_pred[0,0]:.2f}) - FAILED TO LEARN")
            else:
                print(f"   ? Shortcut behavior unclear (x={sc_pred[0,0]:.2f}, vx={sc_pred[0,2]:.2f})")
            
            if pos_diff < 0.1:
                print(f"\n   ⚠️  WARNING: Models predict IDENTICAL positions!")
                print(f"      This suggests models learned the SAME thing")
            elif pos_diff > 0.5:
                print(f"\n   ✅ Models predict DIFFERENT positions!")
                print(f"      Differentiation gap: {pos_diff:.2f}")
    
    # Final diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    
    # Check largest horizon
    dt_tensor = torch.FloatTensor([[1.0]])
    with torch.no_grad():
        seq_velocity = seq_model(state, actions, time, dt_tensor)
        seq_pred = state + seq_velocity * dt_tensor
        sc_velocity = shortcut_model.velocity_net(state, actions, time, dt_tensor)
        sc_pred = state + sc_velocity * dt_tensor
    
    pos_diff = abs(seq_pred[0,0].item() - sc_pred[0,0].item())
    
    print(f"\nAt largest horizon (dt=1.0):")
    print(f"  Position difference: {pos_diff:.4f}")
    
    if pos_diff < 0.1:
        print(f"\n❌ MODELS ARE IDENTICAL")
        print(f"   Both models predict the same thing at large timesteps")
        print(f"   ")
        print(f"   POSSIBLE CAUSES:")
        print(f"   1. Training bug - models weren't differentiated during training")
        print(f"   2. Dataset bug - collision scenarios not diverse enough")
        print(f"   3. Architecture bug - models can't learn complex collision dynamics")
        print(f"   4. Loss function bug - bootstrap hierarchy not working")
        print(f"   ")
        print(f"   RECOMMENDATION: Need to investigate training code")
        
    elif pos_diff > 0.5:
        print(f"\n✅ MODELS ARE DIFFERENT")
        print(f"   Models make different predictions at large timesteps!")
        print(f"   ")
        print(f"   This means:")
        print(f"   - Training worked correctly ✓")
        print(f"   - Models learned different behaviors ✓")
        print(f"   - Evaluation code likely has bug ✓")
        print(f"   ")
        print(f"   RECOMMENDATION: Fix evaluation code in evaluate_collision_physics.py")
        print(f"   The bug is in predict_shortcut_single_step() method (lines 151-165)")
        
    else:
        print(f"\n⚠️  UNCLEAR")
        print(f"   Small difference detected ({pos_diff:.4f})")
        print(f"   May need more investigation")

if __name__ == "__main__":
    main()