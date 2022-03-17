# Every model gets data, produces grads
# These grads are synced - every model has identical grads now (verified in the single_device tests)
# the models are identical, the grads are identical, the states are identical (since they have only been initialized yet)
# therefore, the updated model, and states should be identical - test this
