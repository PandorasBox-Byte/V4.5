"""Core package init for V4.5 project.

Compatibility shim: some versions of `huggingface_hub` removed `cached_download`.
Provide a small alias to `hf_hub_download` if available so older imports keep working.
"""

try:
	import huggingface_hub as _hfh
	if not hasattr(_hfh, "cached_download"):
		# prefer top-level hf_hub_download, otherwise try utils
		target = getattr(_hfh, "hf_hub_download", None)
		if target is None:
			try:
				from huggingface_hub.utils import hf_hub_download as target
			except Exception:
				target = None

		if target is not None:
			def cached_download(*args, **kwargs):
				# map simple signature to hf_hub_download where possible
				return target(*args, **kwargs)

			setattr(_hfh, "cached_download", cached_download)
except Exception:
	# don't prevent imports if huggingface_hub not installed
	pass
