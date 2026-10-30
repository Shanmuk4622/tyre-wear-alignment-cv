# Training-only ATLAS appearance/reference gate

All 386 source images retain role=train. V1 query/support photos are reported
separately. Seven fixed probes use the frozen selected epoch18 classifier;
there is no optimizer, new training, validation/test query or new independent
tyre. New local-contrast channels are a deterministic feature proposal, not a
trained replacement classifier or a depth map. The same frozen v1 image is
canonicalized once; transformations occur after its predicted atlas.

The proposed bank uses original training photos only, one case per distinct
photo, least-represented available physical tyre first, then fixed descriptor
diversity/coverage/glare score. It must retain at least one original query photo
per known training tyre. Low texture is flagged, not excluded as presumed worn
tread. Chroma/glare heuristics are not human annotations or causal attribution.
Candidate pixel arrays are direct .npy and explicitly UNTRAINED. Do not install
them in the v1 classifier whose coefficients were trained against a different bank.

Local contrast: channel-mean luminance; masked 9×9 local mean/std; std floor .02;
clip standardized luminance to ±3 and divide by3; Sobel x/y and gradient magnitude.
Relative L2 is change/original feature norm, not accuracy. Input clipping, the
std floor and flat regions limit scale equivariance. The unclipped supported
subset and denominators are explicit in feature_invariance.csv.

Raw CSVs and native original RGB/valid/grid, dense predictions and descriptors
are retained. Directory names epoch_0000… come from the reused cursor store:
they are blocks of50 input images, not training epochs. Every image is one
atomic recovery unit coupled to CSV byte/row cursors and native artifacts.
Candidate assembly is a separate final unit; interruption resumes without
changing the frozen source/environment/weights. Actual source hashes and role
registry are included. Native .pt/.npy are direct; no data ZIP/NPZ is uploaded.

The source fit and workstations remain untouched. Better consistency of these
fixed features does not establish unseen-tyre classification, calibration,
physical wear/depth/alignment or safety. Known original train tyre support is
still only 1/3/4 by class; video parent mapping remains unresolved. Any future
model/data/input/bank/objective/sampler change requires a new frozen protocol.
