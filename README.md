# ligandmpnn-jax

JAX port of LigandMPNN model, currently work in progress.

The original ProteinMPNN/LigandMPNN implementation and papers can be found at:

[Robust deep learning–based protein sequence design using ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) - Dauparas, Justas, et al. "Robust deep learning–based protein sequence design using ProteinMPNN." Science 378.6615 (2022): 49-56.

[Atomic context-conditioned protein sequence design using LigandMPNN](https://www.nature.com/articles/s41592-025-02626-1) - Dauparas, Justas, et al. "Atomic context-conditioned protein sequence design using LigandMPNN." Nature Methods 22.4 (2025): 717-723.

[LigandMPNN Source Code](https://github.com/dauparas/LigandMPNN)

So far, ProteinMPNN-portion is ported, but need to add
the ligand graph support as well.

I've changed the way the configuration and how data is passed around the code to make it a bit easier to follow, otherwise the functionality should remain the same for the most part.

- [x] ProteinMPNN
- [ ] LigandMPNN
- [ ] Membrane models
- [ ] Packing