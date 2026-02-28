# Brain Monitor - Neural Activity Visualization

The Brain Monitor provides a real-time visualization of EvoAI's internal code execution, showing which core modules are actively "firing" like neurons in a brain.

## Features

- **Live ASCII Visualization**: Tree view of all core/*.py files
- **Activity Tracking**: Files light up GREEN when their code executes
- **Statistics Panel**: Shows active module count, total fires, and top firing neurons
- **Separate Terminal Window**: Runs alongside the main EvoAI interface (macOS)
- **Real-time Updates**: 20 FPS refresh rate with 0.3s flash duration

## Usage

### Automatic Launch (Default)

The brain monitor launches automatically when you start EvoAI:

```bash
./Start_Engine.sh
```

A second Terminal window will open showing the brain monitor visualization.

### Disable Brain Monitor

To disable the automatic launch:

```bash
export EVOAI_ENABLE_BRAIN_MONITOR=0
./Start_Engine.sh
```

### Standalone Testing

To run the brain monitor by itself:

```bash
python3 core/brain_monitor.py
```

Press `q` to quit the visualization.

### Test Integration

To verify the brain monitor is working:

```bash
python3 test_brain_monitor.py
```

## Visualization Key

- **Green ⚡**: File is actively executing (neuron firing)
- **White**: File is inactive
- **Cyan**: Headers and section titles
- **Yellow**: Statistics and metrics

## How It Works

The brain monitor uses Python's `sys.settrace()` to intercept function calls within the core/ directory. When code from a file executes, it "fires" (flashes green) for 0.3 seconds. The system tracks:

- **Active modules**: Files currently executing
- **Total neural fires**: Cumulative execution count
- **Top firing neurons**: Most frequently accessed files

This provides insight into EvoAI's "thought process" - which modules are being used and how often.

## Technical Details

- **Tracking Method**: `sys.settrace()` with minimal performance overhead
- **UI Framework**: curses for terminal graphics
- **Threading**: Thread-safe with locks for concurrent access tracking
- **Platform**: macOS (uses AppleScript to launch separate Terminal window)
- **Update Rate**: 20 FPS (50ms delay between frames)
- **Flash Duration**: 300ms per activity event
- **Activity Log**: Rolling buffer of last 1000 events

## Environment Variables

- `EVOAI_ENABLE_BRAIN_MONITOR`: Enable/disable (default: `1`)

## Files

- `core/brain_monitor.py`: Main implementation
- `test_brain_monitor.py`: Standalone test script
- Integration points:
  - `core/launcher.py`: Launches monitor window at startup
  - `core/engine_template.py`: Installs trace hook in Engine.__init__
