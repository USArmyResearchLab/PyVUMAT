# Python Vectorized User MATerial (PyVUMAT) Framework

The Python Vectorized User Material (PyVUMAT) framework enables users to develop custom material models in Python and seamlessly integrate them into finite element analysis simulations. Development in Python can significantly reduce development time compared to compiled languages (i.e., C, C++, and Fortran) by leveraging the vast collection of freely available Python packages. This is especially true for the emerging field of machine learning (ML)-based material models, where established Python packages such as PyTorch, TensorFlow, and JAX can drastically simplify ML model development. Direct use of these packages will allow ML material model development to keep pace with the rapid advancements in the ML community. PyVUMAT may also lower the barrier to creating novel material models for researchers who are unfamiliar with C, C++, or Fortran.

## Documentation

Instructions to build and run PyVUMAT are provided in the [User's Guide](UsersGuide_PyVUMAT_2.0.pdf). The guide also provides guidance on how users can create custom VUMAT models in Python using PyVUMAT. Release notes describing changes made for version 2.0 are also provided.

## Citations

If PyVUMAT has been significant to your research, please cite:

-  Crone, Joshua C. "PyVUMAT: A package to develop and deploy machine learning material models in finite element analysis simulations." Computational Materials Science 262 (2026): 114377.

In bibtex format:

```
@article{crone2026pyvumat,
  title={PyVUMAT: A package to develop and deploy machine learning material models in finite element analysis simulations},
  author={Crone, Joshua C},
  journal={Computational Materials Science},
  volume={262},
  pages={114377},
  year={2026},
  publisher={Elsevier}
}
```

## License

PyVUMAT is licensed under the Creative Commons Zero 1.0 Universal (CC0 1.0) license. Please see [LICENSE](LICENSE) for details.
