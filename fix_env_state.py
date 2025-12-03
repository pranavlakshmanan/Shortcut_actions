#!/usr/bin/env python3
"""Fix the env.state bug in all scripts"""

import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: Fix "env.state = state.copy()" in simulate functions
    content = re.sub(
        r'env\.state = state\.copy\(\)',
        '''env.reset()
    env.particles[0]["position"] = state[:2].astype(np.float32)
    env.particles[0]["velocity"] = state[2:].astype(np.float32)''',
        content
    )
    
    # Pattern 2: Fix "self.env.state = initial_state.copy()"
    content = re.sub(
        r'self\.env\.state = initial_state\.copy\(\)',
        '''self.env.reset()
        self.env.particles[0]["position"] = initial_state[:2].astype(np.float32)
        self.env.particles[0]["velocity"] = initial_state[2:].astype(np.float32)''',
        content
    )
    
    # Pattern 3: Fix "return env.state.copy()" -> use next_state
    content = re.sub(
        r'return env\.state\.copy\(\)',
        'return next_state.copy()',
        content
    )
    
    # Pattern 4: Fix "return self.env.state.copy()"
    content = re.sub(
        r'return self\.env\.state\.copy\(\)',
        'return next_state.copy()',
        content
    )
    
    # Pattern 5: Fix quick_test.py style
    content = re.sub(
        r'env\.state = state\.cpu\(\)\.numpy\(\)\.squeeze\(\)',
        '''state_np = state.cpu().numpy().squeeze()
    env.reset()
    env.particles[0]["position"] = state_np[:2].astype(np.float32)
    env.particles[0]["velocity"] = state_np[2:].astype(np.float32)''',
        content
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes: {filepath}")

# Fix all relevant files
fix_file('state_based/train_state_shortcut.py')
fix_file('state_based/evaluate_state.py')
fix_file('state_based/quick_test.py')
fix_file('verify_results.py')

print("\nDone! Now re-run training and evaluation.")
