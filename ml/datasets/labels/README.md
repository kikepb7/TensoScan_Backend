# Labels of the real datasets

Only text lives here; the photos stay out of git (C-ML-03).

## ds-real-0.1.0 (Medisana, one device)

- Source: 251 photos of one Medisana monitor in a shared Drive folder. Files are renamed `pNNN.jpg` in order of
  their original name; `ds-real-0.1.0.photo-map.csv` maps each one back to the original name.
- `ds-real-0.1.0.labels.csv`: 235 photos with SYS/DIA/PUL **read from the images by Claude, not yet verified by the
  owner** (53 distinct readings), plus the display crop `roi_*` found from the green START/STOP button (hand
  corrected for 11 photos).
- `ds-real-0.1.0.needs-review.csv`: 15 photos left out because the value was unclear, the monitor was strongly
  rotated or the range looks implausible (SYS 177 / DIA 157). They need a human reading.
- The split is by reading (one device): train 30, val 11, test 12 readings. The `beta` channel only.
