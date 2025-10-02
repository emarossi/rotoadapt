<h1> ADAPT project </h1>
<p>
Algorithms implementing energy-based (Rotoselect) and gradient-based strategies to select generators in an ADAPT-VQE algorithm.<br>
The Rotoselect algorithm is based on the ExcitationSolve algorithm -> https://arxiv.org/abs/2409.05939<br>
The Gradient-based ADAPT is implemented - based on the work of Anurag Singh (https://github.com/darkcoordinate/SlowQuant/blob/master/slowquant/unitary_coupled_cluster/sa_adapt_wavefunction.py#L281) - according to the original ADAPT-VQE paper -> https://doi.org/10.1038/s41467-019-10988-2<br> 
</p>
<h3> Dependendencies </h3>
Besides the standard python numerical libraries you need
<ol>
 <li> Slowquant: https://github.com/erikkjellgren/SlowQuant.git </li>
 <li> PySCF </li>
 <li> argparse, pickle </li>
  <li> multiprocessing </li>
</ol>
<h3> How to launch a calculation </h3>
Call the '*_production.py' as in the example to calculate LiH, (4,4) active space with rotoselect use

```
python rotoadapt_production.py --mol --AS 4 4
```
Call the option --help to visualize available options. Only --mol and --AS need be specified for chemical accuracy calculations.