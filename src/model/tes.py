from WeightInitialization import WeightInitiatior,WeightInitMethod

# initiator = WeightInitiatior(WeightInitMethod.ZERO, neurons=2, layers=3)
# print(initiator.get_weights())
# initiator = WeightInitiatior(WeightInitMethod.UNIFORM, neurons=2, layers=3, lower_bound=4, upper_bound=7)
# print(initiator.get_weights())
initiator = WeightInitiatior(init_method=WeightInitMethod.NORMAL, neurons=2, layers=3, mean=24, std=2, seed=69)
weights,bias = initiator.get_weights()
print(weights)
print("*"*50)
print(bias)