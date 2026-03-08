from .family import SDXLFamily as SDXLFamily
from .trainer import SDXLTrainer as SDXLTrainer
from .loader import SDXLLoader as SDXLLoader

# This __init__ makes the family's primary components accessible.
# The ModelRegistry auto-discovers ModelFamily subclasses via introspection,
# so explicit registration is not needed here.
