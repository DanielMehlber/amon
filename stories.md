# User Stories

## Start monitoring session

**As an operator**, I want to start the monitoring session by invoking a shell program that loads the configuration file and begins to process the configured video stream.

### Acceptance Criteria

- The configured video source is opened.
- Errors are shown if the source cannot be opened.

---

## Automatic calibration

**As an operator**, I want the system to perform an automatic calibration phase before monitoring starts so that no manual threshold tuning is required.

### Acceptance Criteria

- Calibration duration is configurable.
- Calibration starts automatically.
- Calibration determines detector-specific thresholds.
- Monitoring starts automatically after calibration.

---

## Long-running monitoring

**As an operator**, I want monitoring to run continuously for many days so that prolonged observations require no supervision.

### Acceptance Criteria

- Monitoring continues until shell application is stopped.
- No user interaction is required after startup.
- Monitoring remains responsive while events are recorded.

---

## Detect temporal anomalies

**As an operator**, I want temporal disturbances detected automatically.

### Acceptance Criteria

- Noise is detected.
- Flickering is detected.
- Contrast changes are detected.
- Detector thresholds originate from calibration.

---

## Detect HUD anomalies

**As an operator**, I want changes in HUD elements detected so that device status irregularities are reported.

### Acceptance Criteria

- Text changes are detected.
- Blink frequency changes are detected.
- Blink start/stop is detected.
- Position changes are detected.
- Size changes are detected.

---

## Detect spatial anomalies

**As an operator**, I want distortions of the scene background detected so that image corruption is reported.

### Acceptance Criteria

- HUD elements are ignored.
- Spatial distortions create events.

---

## Reduce false positives

**As an operator**, I want overlapping anomalies classified correctly so that only the primary anomaly is reported.

### Acceptance Criteria

- Hierarchical suppression prevents duplicate events.
- Related anomalies are merged where appropriate.

---

## Record anomaly events

**As an operator**, I want each anomaly recorded as a single event so that reports remain concise.

### Acceptance Criteria

- One event is created per anomaly occurrence.
- Start time is recorded.
- End time is recorded.
- Duration is calculated automatically.
- Calibrated threshold is used.

---

## Store event metadata

**As an operator**, I want additional information stored for every anomaly so that later analysis is possible.

### Acceptance Criteria

- Maximum intensity is stored.
- Intensity timeline is stored.
- Detector-specific metadata is stored.

---

## Capture visual evidence

**As an operator**, I want every anomaly accompanied by a GIF or video sample so that I can verify detections and replay them.

### Acceptance Criteria

- Media is generated automatically.
- Long events are clipped to a configurable duration.
- Spatial anomalies highlight the affected region.

---

## Inspect reports

**As an operator**, I want to inspect monitoring results in a browser so that I can review anomalies conveniently.

### Acceptance Criteria

- Pending reports are viewable.
- Completed reports are viewable.
- Events are listed chronologically.
- Events can be filtered for a quicker search.

---

## Inspect calibration

**As an operator**, I want to review calibration results so that I can verify successful calibration.

### Acceptance Criteria

- Calibration video/GIF is shown.
- Feature points are visualized.
- HUD markers are shown.
- HUD text is annotated.
- Blink frequencies are annotated.

---

## Inspect event details

**As an operator**, I want detailed information for every event so that I can evaluate anomalies.

### Acceptance Criteria

Each event displays:

- Anomaly type
- Start and end timestamps
- Duration
- GIF/video sample
- Intensity plot
- Threshold line
- Detector metadata

---


## Export reports

**As an operator**, I want reports exportable so that they can be archived or emailed.

### Acceptance Criteria

- Export as HTML archive.
- Support additional export formats in the future.

---

## Fully offline operation

**As an operator**, I want the application to work completely offline so that it can be used in isolated environments.

### Acceptance Criteria

- No internet connection required.
- No cloud services required.

---

## Persistent storage

**As an operator**, I want anomaly events stored locally so that previous monitoring sessions can be reviewed later.

### Acceptance Criteria

- Uses a file-based database.
- Events persist across application restarts.

# Developer Stories

## Detector interface

Implement a detector interface.

### Acceptance Criteria

- Supports calibration mode.
- Supports detection mode.
- Receives image frames.
- Outputs a mapping of anomaly IDs to intensities.
- Allows detector-specific configuration.

---

## Detector loading

Implement detector plugin loading.

### Acceptance Criteria

- Detectors are loaded from configuration.
- New detectors require no application code changes.

---

## Calibration lifecycle

Implement the detector lifecycle.

### Acceptance Criteria

- Calibration phase.
- Threshold estimation.
- Switch to detection phase.

---

## Video source abstraction

Implement a generic video source interface.

### Acceptance Criteria

- Supports frame iteration.
- Easily extended by additional source types.

---

## File source

Implement a video file source.

### Acceptance Criteria

- Reads frames sequentially.
- Configurable input file.

---

## Real-time processing pipeline

Implement the monitoring pipeline.

### Acceptance Criteria

- Real-time processing.
- Non-blocking architecture.
- Queue-based communication with background workers and I/O operations that would otherwise stall the pipeline.

---

## Event aggregation

Merge frame detections into anomaly events.

### Acceptance Criteria

- Event starts when threshold is exceeded.
- Event ends when intensity falls below threshold.
- Continuous anomalies produce exactly one event.

---

## Anomaly hierarchy

Implement anomaly precedence.

### Acceptance Criteria

- Prevent duplicate reporting.
- Higher-priority anomalies suppress lower-priority ones.

---

## Local database

Implement persistent event storage.

### Acceptance Criteria

- Stores timestamps.
- Stores metadata.
- Stores media references.

---

## Background writer

Move all blocking I/O into background workers.

### Acceptance Criteria

- Database writes are asynchronous.
- GIF/video generation is asynchronous.
- Monitoring pipeline never blocks on I/O.

---

## Event media generation

Generate visual evidence for events.

### Acceptance Criteria

- Creates GIF or video.
- Clips long-running events.
- Highlights regions of interest when applicable.

---

## Calibration media

Generate annotated calibration media.

### Acceptance Criteria

- Shows detected keypoints.
- Shows HUD locations.
- Annotates HUD text.
- Displays blink frequencies.

---

## Report UI

Implement the report interface using Panel.

### Acceptance Criteria

- White theme.
- Responsive layout.
- No handwritten HTML.

---

## Event visualization

Implement event detail views.

### Acceptance Criteria

- Media preview.
- Intensity plot.
- Threshold visualization.
- Metadata display.
- Highlighted regions for spatial anomalies.

---

## Report export

Implement report export.

### Acceptance Criteria

- Standalone HTML archive export.

---

## Configuration system

Externalize configuration.

### Acceptance Criteria

- Calibration duration.
- Detector list.
- Video source.
- Export settings.
- Detector configuration.

---

## Synthetic test video

Generate a synthetic test video for testing purposes that contains extensive and overlapping anomalies.

### Acceptance Criteria

- Contains every anomaly type.
- Includes overlapping anomalies.
- Uses known timestamps for verification.

---

## End-to-end pipeline tests

Implement full integration tests.

### Acceptance Criteria

- Verifies event creation.
- Verifies timestamps.
- Verifies durations.
- Verifies database contents.
- Verifies generated media.

---

## Detector unit tests

Implement detector unit tests.

### Acceptance Criteria

- Calibration verified.
- Threshold behavior verified.
- Edge cases covered.

---

## User documentation

Create `for-users.md`.

### Acceptance Criteria

- Installation.
- Configuration.
- Running monitoring.
- Viewing reports.
- Exporting reports.

---

## Developer documentation

Create `for-dev.md`.

### Acceptance Criteria

- Detector interfaces.
- Pipeline architecture.
- Image-processing algorithms.
- Design rationale.
- Plugin architecture.
- Technical decisions.

---

## Maintainability

Improve code quality.

### Acceptance Criteria

- Public APIs documented with docstrings.
- Girly code.
- Appropriate explanatory comments.
- Simple, readable code.
- Avoid unnecessary hard-coded values.
- Follow KISS principles.