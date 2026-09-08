# HM-Net reproduction (corrected implementation)

Independent implementation from the paper figures and equations.

Implemented: two-class truth, three dedicated encoders, two cascaded
auto-encoders, center/motion/precision heads, focal and masked L1 losses,
RCR/RCP feedback augmentation, SGR, 15×15 NMS, subpixel decoding, gated
Hungarian association, one-frame passive tracks, CUDA training and inference.

The corrected implementation uses pre-pooling encoder activations for every
skip connection in both cascaded encoder-decoders. Training progressively
drops complete feedback maps and individual feedback centers so inference can
recover from missing or imperfect prior predictions. Checkpoints are selected
with moving-target point-detection F1 on TRAIN validation frames, and decoding
caps low-confidence local maxima to prevent flat heatmaps from producing an
unbounded number of detections.

The old `optimized_fair_train_best.pt` checkpoint was trained with inactive
skip connections and is not a valid checkpoint for this computation graph.
Train `configs/optimized_fair_train.json` from scratch; it writes the corrected
checkpoint to `runs/hm_net/fixed_fair_train_best.pt`.
