Vendored OmniStream encoder (https://github.com/Go2Heart/OmniStream, MIT; DINOv3 layers under their
own headers), copied unchanged from `causal_forcing_repa_robotics/wan/model`. It is used only as the
frozen REPA target encoder in `../losses.py`; load weights with
`OmnistreamMultiFrameTransformer.from_pretrained(<OmniStream checkpoint dir>)`.
