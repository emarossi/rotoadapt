<h1> Efficient Rotoselect algorithm </h1>
<p>
Implementation of the efficient Rotoselect algorithm described in https://arxiv.org/abs/2606.04786.<br> 
A calculation can be lauched via the 'rotoadapt_production_*.py' scripts (where *_eq.py script considers equilibrium geometries, while *_str.py the stretched geometries). For example, a Rotoselect efficient equilibrium geometry calculation can be run via:
 
```
python rotoadapt_production_eq.py --mol LiH --AS 4 6 --gen --po -oo --eff
```
The flags can be selected according to:
<ul>
  <li> --mol: molecular system (LiH, H2O, BeH2) </li>
  <li> --AS: active space (nEL, nMO) </li>
  <li> --gen: (if specified) use generalized excitation operators </li>
  <li> --po: (if specified) ansatz parameter optimization </li>
  <li> --oo: (if specified) orbital optimization </li>
  <li> --eff: run efficient Rotoselect algorithm, if specified; if not, run ExcitationSolve algorithm (https://doi.org/10.1038/s42005-025-02375-9) </li>
</ul>
The fermionic ADAPT-VQE algorithm (https://doi.org/10.1038/s41467-019-10988-2) can be run via the 'adapt_production_*.py' scripts via:
 
```
python adapt_production_eq.py --mol LiH --AS 4 6 --gen --po -oo --eff
```
Here, apart from '--eff' all the flags of the 'rotoadapt_production_*.py' scripts apply.
</p>
<h3> Dependencies </h3>
The python packages required are enlisted in 'python_env_requirements.txt'. The code requires the installation of a local branch of Slowquant (https://github.com/erikkjellgren/SlowQuant.git). Please, <a href="mailto:manuross95@gmail.com">contact me</a> for more info.
