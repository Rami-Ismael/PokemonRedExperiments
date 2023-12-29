from pyboy import PyBoy
from pyboy.logger import log_level
pyboy = PyBoy(
    "pokemon_red.gb",
    debugging=False,
    disable_input=False,
    window_type = "SDL2",
)
pyboy.botsupport_manager().screen()
pyboy.set_emulation_speed(5)