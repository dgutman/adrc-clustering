def get_float(x):
	""" get_float()
	
	Parse float from string, returns float. If parsing fails
	return the original value

	Arguments: 
		string: some value

	Return
		float or string
	"""
	try:
		return float(x)
	except ValueError:
		return x 
