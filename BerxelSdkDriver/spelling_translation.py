"""Translate corrected Python API symbol names to vendor SDK symbol names."""

# Map corrected API names to the misspelled native symbol names
# currently exported by the vendor .so.
NATIVE_SYMBOL_TRANSLATIONS = {
    "berxelGetDeviceIntrinsicParams": "berxelGetDeviceIntriscParams",
    "berxelGetCurrentStreamFrameMode": "berxelGetCurrentStramFrameMode",
}


def native_symbol_name(api_symbol_name: str) -> str:
    """Return the actual native symbol name for a corrected API symbol."""
    return NATIVE_SYMBOL_TRANSLATIONS.get(api_symbol_name, api_symbol_name)


def bind_native_symbol(library, api_symbol_name: str):
    """Resolve and return a ctypes function pointer from the loaded library."""
    symbol_name = native_symbol_name(api_symbol_name)
    try:
        return getattr(library, symbol_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Missing native symbol '{symbol_name}' (requested API '{api_symbol_name}')."
        ) from exc
