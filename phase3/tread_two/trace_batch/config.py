"""One source/data/environment-bound candidate; disposable gates have another ID."""
from common import HERE,read,digest,environment
from schema import specification

REPO='Shanmuk4622/tyre-wear-study'
NAMESPACE='phase3/tread-two/trace-r1'
GEOMETRY=dict(revision='1113bf8a9dd69687b23570fe985ed3a979a5f0e1',
    sha256='8358e0595d46caee1160818ad975042f2031e14f7d0fa6f2ac105b9312325450',
    url='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/1113bf8a9dd69687b23570fe985ed3a979a5f0e1/phase3/tread-two/geo-r1/fit/7d1bc8554813ae16ba3d73d5d4af2d8c05c977e9282d1bf324ff9dbef6e89bdb/evidence/exports/final-8358e0595d46caee1160.pt',
    selection='unselected_final_geometry_only_original_GEO_selected_None_preserved')

def configuration(fixture=False):
    return dict(version='trace-r1-executable-v1',mode='disposable_engineering' if fixture else 'one_measured_candidate',
        seed=1,epochs=1 if fixture else 60,batch=8,updates=2 if fixture else 2940,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,clip=1.,scheduler=None,scaler=None),
        geometry=GEOMETRY,data=read(HERE/'DATA_PIN.json'),environment=environment(),sources=read(HERE/'FILES.json'),schema=specification(),
        losses=dict(local=1.,masked=1.,pair=.1,variance=.1,covariance=.01,beta=.1,variance_floor=.1,diversity_min_rows=7),
        selection=dict(masked_gain=.05,local_gain=.1,coverage=.95,eligible_primary_fraction=.9,projector_rank=4.,projector_std=.05,
                       local_rank=4.,spatial_rank=4.,within_local_rank=2.,within_spatial_rank=2.,cosine_norm_floor=1e-8),
        storage=dict(logical_ceiling=15000000000,peak_ceiling=18000000000,reserve=2000000000,uncertainty=1.2),
        retention=dict(native_training_tokens=True,full_predictions_every_view=True,target_maps_once=True,epoch_weights='all',ZIP=False,NPZ=False),
        appearance=dict(low=.9,high=1.1),probes=['identity','brightness_0p85','grayscale','reverse_bands','reverse_columns'],
        publish_seconds=1800,workstation_changes=False,new_photos=0)
