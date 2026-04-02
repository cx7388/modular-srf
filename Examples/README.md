# Examples

This folder contains importable JSON configurations for the SRF examples included in this project.

Each file is an `srf-elicitation-config` snapshot that can be loaded through the application's import function. The examples are intended to reproduce published cases within this software environment, either from the original method papers or from the modular SRF paper underlying this project.

## How to use

1. Launch the application.
2. Open the elicitation page.
3. Use the JSON import function.
4. Select one of the files in this folder.
5. Click `Calculate` to reproduce the corresponding case.

## Example files

| File | Method in app | Provenance |
| --- | --- | --- |
| `Example_Predefined SRF.json` | SRF | Example from Huang et al. (2026) and the software paper, implemented through the predefined SRF workflow. |
| `Example_Modular SRF.json` | Modular SRF | Example from Huang et al. (2026) and the software paper, implemented through the modular questionnaire workflow. |
| `Original SRF.json` | SRF | Example based on Figueira and Roy (2002). |
| `SRF-II.json` | Modular SRF configuration reproducing an SRF-II case | Example based on Abastante et al. (2022). Stored as a modular questionnaire import. |
| `Robust SRF.json` | Robust SRF | Example based on Siskos and Tsotsolas (2015). |
| `WAP.json` | WAP | Example based on Tsotsolas et al. (2019). |
| `Imprecise SRF.json` | Imprecise SRF | Example based on Corrente et al. (2017). |
| `Belief-degree Imprecise SRF.json` | Belief-degree Imprecise SRF | Example based on Zhang and Liao (2023). |

## Notes

- The `Example_*` files are taken from the modular SRF paper and the software paper. They are included to compare the predefined and modular workflows on the same case.
- The `SRF-II.json` example is stored with `srf_method = modular_srf` because the published case combines SRF-II and imprecise SRF components.
- Criterion names, rank layouts, and parameter values are organized to match the software implementation of the published examples as closely as possible.

## Sources

- Huang, R., Kadzinski, M., Figueira, J. R., Corrente, S., Siskos, E., & Burgherr, P. (2026). *A Modular Simos-Roy-Figueira Framework for Tailored Weight Elicitation in Multi-Criteria Decision Aiding*. Expert Systems with Applications, 311, 131315. https://doi.org/10.1016/j.eswa.2026.131315
- Figueira, J., & Roy, B. (2002). *Determining the weights of criteria in the ELECTRE type methods with a revised Simos' procedure*. European Journal of Operational Research, 139(2), 317-326. https://doi.org/10.1016/S0377-2217(01)00370-8
- Abastante, F., Corrente, S., Greco, S., Lami, I. M., & Mecca, B. (2022). *The introduction of the SRF-II method to compare hypothesis of adaptive reuse for an iconic historical building*. Operational Research, 22(3), 2397-2436. https://doi.org/10.1007/s12351-020-00611-4
- Siskos, E., & Tsotsolas, N. (2015). *Elicitation of criteria importance weights through the Simos method: A robustness concern*. European Journal of Operational Research, 246(2), 543-553. https://doi.org/10.1016/j.ejor.2015.04.037
- Tsotsolas, N., Spyridakos, A., Siskos, E., & Salmon, I. (2019). *Criteria weights assessment through prioritizations (WAP) using linear programming techniques and visualizations*. Operational Research, 19(1), 135-150. https://doi.org/10.1007/s12351-016-0280-7
- Corrente, S., Figueira, J. R., Greco, S., & Slowinski, R. (2017). *A robust ranking method extending ELECTRE III to hierarchy of interacting criteria, imprecise weights and stochastic analysis*. Omega, 73, 1-17. https://doi.org/10.1016/j.omega.2016.11.008
- Zhang, Z., & Liao, H. (2023). *An evidential reasoning-based stochastic multi-attribute acceptability analysis method for uncertain and heterogeneous multi-attribute reverse auction*. Journal of the Operational Research Society, 74(1), 239-257. https://doi.org/10.1080/01605682.2022.2035271
