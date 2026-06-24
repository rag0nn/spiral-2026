from spiral.struct.packets import *
import pytest
from beartype.roar import BeartypeCallHintParamViolation

def test_translation_object():
    valid_params = (0.1,0.0001,-0.05)
    TranslationObject(*valid_params)
    invalid_params = (1,"123",-0.50)
    with pytest.raises(BeartypeCallHintParamViolation):
        TranslationObject(*invalid_params)
        