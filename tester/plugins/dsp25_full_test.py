"""
Plugin for testing everything on the DSP25 chip
"""
import logging
import random

from utils import *

# Begin logging
LOGGER = logging.getLogger(__name__)

# Custom behavior that can replace ShmooConstantTest
class DSP25FullTest(ShmooTest):
	def create_payload(self):
		return b"", {}
	def check_output(self, context, value):
		return value == "Success!", value

# Define the test and register it
ShmooTestHarness.register_test_suite(TestSuite("dsp25full",
	"build/location/of/elf/file.elf",

	# Currently using ShmooConstantTest bc input/output don't change
	ShmooConstantTest("DSP25 Full: Pre-gen Input, Golden Model Result Check", 0x0, to_chip=b"", expect=b"Success!"),
	# DSP25FullTest("Basic full test", 0x0, timeout=120),
))

# Exports
__all__ = []
