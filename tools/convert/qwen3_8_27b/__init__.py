"""Qwen3.8-27B target adapters: the DSpine draft source contract.

``dspine_recipe`` is the DSpine counterpart of the DFlash contract in
``tools.convert.dflash``: it owns the checkpoint config contract, the object
inventory, and the physical format choice.  The DSpine suffix is attached to an
existing v3 artifact by ``scripts/attach_dspine_head.py`` because its
vocabulary-wide refinement tables are derived from the *target*'s output head.
"""
