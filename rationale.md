# Project Overview

## Rationale

Monitoring long-running video streams is a tedious and error-prone task when performed manually. Engineers currently need to observe a static camera feed for extended periods to identify visual anomalies, often over the course of several days. This process is expensive, inefficient, and susceptible to human error caused by fatigue or inattention.

The purpose of this software is to automate this monitoring process by continuously analyzing a video stream in real time, detecting anomalies, and recording them as meaningful events. Rather than reporting every affected frame, the system groups continuous occurrences into a single event with a clearly defined start time, end time, duration, and supporting evidence. This allows engineers to review only the relevant incidents after a monitoring session has completed.

Beyond being an application, the software is intended to serve as a lightweight framework that allows engineers to extend its functionality by implementing new anomaly detectors and video sources without modifying the core monitoring pipeline. Since interface implementations should be concise, all generic logic should be handled by the framework, not the interface implementations. Separation of concerns is key.

---

# High-Level Overview

## Overview

The application continuously monitors a video stream of a static scene overlayed with HUD elements. Before monitoring begins, the system performs an automatic calibration phase that learns the normal characteristics of the video stream, including detector-specific tolerances, stable feature points, HUD elements, and blinking behavior. These measurements establish the baseline against which future observations are compared. For this reason, all detector implementations are set to calibration mode, which forces them to gather statistics and determine thresholds. They are switched back to detection mode after the calibration phase.

After calibration, the monitoring pipeline operates continuously in real time. Individual detectors analyze each incoming frame and calculate intensity values for each anomaly. When those intensities go beyond the automatically calibrated threshold, an anomaly is detected. These detections are aggregated into higher-level anomaly events that begin when detector intensity exceeds the calibrated threshold and end once it returns below the threshold. Continuous anomalies therefore produce exactly one logged event regardless of their duration. This event aggregation is part of the pipeline and executed at the end when all intensity values are emitted. Therefore, the pipeline must query the calibrated threshold from the detector implementation.

For every detected event, the system records timestamps, duration, detector metadata, intensity measurements, and visual evidence such as GIFs or video clips. All event data is stored in a local file-based database for later inspection. This logic is not executed in the pipeline, but on separate processes to avoid stalling and keep the detector implementation focused on their algorithms. 

A browser-based reporting interface allows operators to review completed monitoring sessions, inspect calibration results, filter anomaly events, visualize detector outputs, and export reports in portable formats such as PDF or HTML archives.

The application is designed to operate entirely offline and to support future extensions through configurable detector and video source plugins. Hard-coded values should be avoided and the entire framework should be adaptable using a configuration file. This file should also provide information which detectors to load into the pipeline and how they are configured, rendering extensions to the framework straightforward, without modifying the core functionality.

---

# Functional Requirements

## Monitoring

* Monitor a continuous video stream in real time. 
* Support long-running monitoring sessions lasting multiple days.
* Perform automatic calibration before monitoring begins.
* Transition automatically from calibration into monitoring after a certain duration.

## Calibration

During calibration the system shall automatically determine:

* Detector-specific thresholds and tolerances. For this reason, the detectors should support a calibration mode by their interface.
* Stable tracking points of the background.
* HUD element locations.
* HUD text content.
* HUD blinking frequencies.
* Any additional detector-specific baseline information.

No manual threshold tuning should be required. As little as possible algorithm hyperparameter configuration should be required as well.

## Anomaly Detection

The system shall detect three classes of anomalies.

### Temporal anomalies

* Noise
* Flickering
* Contrast changes

### HUD anomalies

* Text changes
* Blink frequency changes
* Blink start/stop
* Position changes
* Size changes

### Spatial anomalies

* Distortions of the background scene while ignoring HUD overlays.

The architecture shall support adding additional anomaly detectors without modifying the core application.

## Event Aggregation

The system shall:

* Aggregate consecutive frame detections into a single anomaly event. For HUD elements, events needs to be differentiated for each one.
* Record event start and end timestamps.
* Calculate event duration.
* Avoid duplicate reporting of the same anomaly.
* Apply detector precedence to reduce false positives.

It is important to establish a configurable hierarchy of exclusion for events to avoid false positives. For instance, a flickering screen would also trigger various other events, which would be false positives. The event aggregator needs to consider this to capture the core event and not its side-effects. 

## Event Recording

Each anomaly event shall store:

* Anomaly type.
* Start timestamp.
* End timestamp.
* Duration.
* Detector metadata.
* Intensity measurements.
* Visual evidence (GIF or video).

## Reporting

The reporting interface shall provide:

* Overview of completed and running monitoring sessions.
* Event list.
* Event filtering.
* Event detail pages.
* Calibration review.
* Intensity plots.
* Embedded GIF/video previews.
* Export functionality.

## Export

Reports shall be exportable as a standalone HTML archive. The export architecture should allow additional formats to be added later.

## Persistence

The system shall persist monitoring data using a local file-based database.

Stored information includes:

* Event metadata.
* Timestamps.
* Detector outputs.
* References to generated media.

## Extensibility

The application shall function as a lightweight framework supporting:

* Pluggable anomaly detectors.
* Pluggable video sources.
* Configuration-based plugin loading.

---

# Non-Functional Requirements

## Performance

* The monitoring pipeline shall operate in real time.
* Blocking I/O operations shall not interrupt frame processing.
* Database writes and media generation shall execute asynchronously.

## Offline Operation

* The application shall function without internet access.
* No cloud services or external infrastructure shall be required.

## Configurability

* Hard-coded values should be avoided.
* System behavior should be configurable where practical.
* Detector thresholds should preferably be learned during calibration rather than configured manually.
* Hyperparameters of algorithms should be ideally determined during calibration phase if possible, keeping the need for tweaking values in the configuration file minimal.

## Maintainability

* Follow KISS principles.
* Keep interfaces small and well-defined.
* Produce readable code intended for modification by engineers. Write girly code which is concise.
* Document all public APIs.
* Include explanatory comments where beneficial.
* Describe chosen algorithms and technical details for future engineers in an attached mark-down file and in the class docstrings. 

## Testability

The project shall be developed test-first by providing:

* A synthetically generated video containing every anomaly type, allowing for extensive testing.
* Overlapping anomaly scenarios.
* End-to-end integration tests.
* Unit tests for detector implementations.

## Usability

* The reporting interface shall be browser-based.
* The interface shall use a clean white theme.
* Reports should present information visually through media, plots, and annotations.
* Calibration results shall be inspectable to verify detector quality. HUD elements should be marked and annotated.

---

# Technical Constraints

* Operate fully offline.
* Use Panel for the reporting interface.
* Use a local file-based database.
* Move blocking operations into background processes using queues.
* Generate reports asynchronously.
* Support configuration-driven plugin loading.
* Support detector calibration and detection modes.
* Initially implement a video file source.
* Provide user documentation (`for-users.md`).
* Provide developer documentation (`for-dev.md`).
* Design for extensibility without increasing complexity unnecessarily.
