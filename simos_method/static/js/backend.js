let simos_calc_results = {};
let asi_value = null;
let calculationProgressTimer = null;
let calculationProgressRequestInFlight = false;


function ensureCalculationProgressElements() {
    const container = document.getElementById('loader-container');
    if (!container) return null;

    let panel = document.getElementById('calculation-progress-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'calculation-progress-panel';
        Object.assign(panel.style, {
            position: 'absolute',
            left: '0',
            right: '0',
            bottom: '0',
            padding: '0 1rem 0.6rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.35rem',
            alignItems: 'center',
            pointerEvents: 'none'
        });

        const text = document.createElement('div');
        text.id = 'calculation-progress-text';
        Object.assign(text.style, {
            fontSize: '0.88rem',
            color: '#566170',
            textAlign: 'center'
        });

        const bar = document.createElement('div');
        bar.id = 'calculation-progress-bar';
        Object.assign(bar.style, {
            width: 'min(28rem, 92%)',
            height: '0.45rem',
            background: '#e3e8ef',
            borderRadius: '999px',
            overflow: 'hidden'
        });

        const fill = document.createElement('div');
        fill.id = 'calculation-progress-fill';
        Object.assign(fill.style, {
            width: '0%',
            height: '100%',
            background: 'linear-gradient(90deg, #325d88 0%, #4f86b5 100%)',
            borderRadius: '999px',
            transition: 'width 180ms ease'
        });

        bar.appendChild(fill);
        panel.append(text, bar);
        container.appendChild(panel);
    }

    return panel;
}


function updateCalculationProgressDisplay(progress = {}) {
    ensureCalculationProgressElements();
    const container = document.getElementById('loader-container');
    const text = document.getElementById('calculation-progress-text');
    const fill = document.getElementById('calculation-progress-fill');
    if (!container || !text || !fill) return;

    container.style.height = '140px';
    text.textContent = progress.message || 'Running SRF calculation...';

    const numericPercent = Number(progress.percent);
    if (Number.isFinite(numericPercent)) {
        fill.style.width = `${Math.max(4, Math.min(100, numericPercent))}%`;
        fill.style.opacity = '1';
    } else {
        fill.style.width = '18%';
        fill.style.opacity = '0.65';
    }
}


async function pollCalculationProgressOnce() {
    if (calculationProgressRequestInFlight) return;
    calculationProgressRequestInFlight = true;
    try {
        const response = await fetch('/data/calculation_progress.json', { cache: 'no-store' });
        if (!response.ok) return;
        const progress = await response.json();
        updateCalculationProgressDisplay(progress);
        if (progress?.done) {
            stopCalculationProgressPolling();
        }
    } catch (error) {
        console.error('Progress polling error:', error);
    } finally {
        calculationProgressRequestInFlight = false;
    }
}


function startCalculationProgressPolling() {
    stopCalculationProgressPolling();
    updateCalculationProgressDisplay({
        message: 'Preparing SRF calculation...',
        percent: 0
    });
    calculationProgressTimer = window.setInterval(pollCalculationProgressOnce, 450);
    pollCalculationProgressOnce();
}


function stopCalculationProgressPolling() {
    if (calculationProgressTimer) {
        window.clearInterval(calculationProgressTimer);
        calculationProgressTimer = null;
    }
}


// Belief-degree rows are rendered dynamically in the DOM. Read them back into a
// normalized array first, then flatten them later into the backend payload format.
function collectBeliefRowsFromDom(scope, rank = null) {
    const prefix = scope === 'z' ? 'z-value-' : `e-value-${rank}-`;
    const rows = [];

    document.querySelectorAll(`input[id^='${prefix}']`).forEach(input => {
        const suffix = scope === 'z'
            ? input.id.split('z-value-')[1]
            : input.id.split(`e-value-${rank}-`)[1];
        const betaInputId = scope === 'z' ? `z-beta-${suffix}` : `e-beta-${rank}-${suffix}`;
        const betaInput = document.getElementById(betaInputId);
        if (!betaInput) return;

        const value = parseFloat(input.value);
        const beta = parseFloat(betaInput.value);
        rows.push({value, beta});
    });

    return rows;
}


function mergeBeliefRows(rows, integerValues = false) {
    const merged = new Map();

    rows.forEach(row => {
        if (!Number.isFinite(row.value) || !Number.isFinite(row.beta)) return;
        const value = integerValues ? Math.round(row.value) : row.value;
        merged.set(value, (merged.get(value) || 0) + row.beta);
    });

    return Array.from(merged.entries())
        .map(([value, beta]) => ({value: Number(value), beta: Number(beta)}))
        .sort((a, b) => a.value - b.value);
}


function serializeBeliefInputsFromDom() {
    const zValue = {};
    const eValue = {};

    // Collapse repeated values before serializing so the backend receives one
    // probability mass per unique z/e value.
    const zRows = mergeBeliefRows(collectBeliefRowsFromDom('z'), false);
    const zSum = zRows.reduce((acc, row) => acc + row.beta, 0);

    if (!zRows.length) {
        throw new Error("Global z belief distribution requires at least one row.");
    }
    if (zRows.some(row => row.value <= 1 || row.beta <= 0)) {
        throw new Error("Each z row must satisfy z > 1 and beta > 0.");
    }
    if (Math.abs(zSum - 1.0) > 1e-4) {
        throw new Error("The sum of probabilities of different z-values must be exactly 1.0.");
    }

    zRows.forEach((row, idx) => {
        const i = idx + 1;
        zValue[`z-value-${i}`] = row.value;
        zValue[`z-beta-${i}`] = row.beta;
    });

    for (let rank = 1; rank < num_ranks; rank++) {
        const rows = mergeBeliefRows(collectBeliefRowsFromDom('e', rank), true);
        if (!rows.length) {
            throw new Error(`Gap rank ${rank}->${rank + 1} requires at least one belief row.`);
        }

        const eSum = rows.reduce((acc, row) => acc + row.beta, 0);
        if (rows.some(row => !Number.isInteger(row.value) || row.value < 0 || row.beta <= 0)) {
            throw new Error(`Gap rank ${rank}->${rank + 1} requires e >= 0 integer and beta > 0.`);
        }
        if (Math.abs(eSum - 1.0) > 1e-4) {
            throw new Error(`The sum of probabilities of blank cards between rank ${rank} and rank ${rank + 1} must be exactly 1.0.`);
        }

        rows.forEach((row, idx) => {
            const i = idx + 1;
            eValue[`e-value-${rank}-${i}`] = row.value;
            eValue[`e-beta-${rank}-${i}`] = row.beta;
        });
    }

    return {zValue, eValue};
}


function collectOptionalConstraintsFromDom() {
    const dictatorshipEnabled = Boolean(document.getElementById('opt-enable-dictatorship')?.checked);
    const minWeightEnabled = Boolean(document.getElementById('opt-enable-minweight')?.checked);

    return {
        dictatorship: {
            enabled: dictatorshipEnabled
        },
        minimum_weight: {
            enabled: minWeightEnabled,
            value: parseFloat(document.getElementById('opt-minweight-value')?.value || '0')
        }
    };
}


function collectInconsistencySuggestionCountFromDom() {
    const rawValue = parseInt(document.getElementById('opt-inconsistency-suggestions')?.value, 10);
    if (!Number.isInteger(rawValue)) return 3;
    return Math.max(1, Math.min(rawValue, 20));
}


function clearInconsistencyReport() {
    const panel = document.getElementById('inconsistency-report');
    if (panel) panel.remove();
}


function typesetMathInElement(element) {
    if (!element || !window.MathJax || typeof window.MathJax.typesetPromise !== 'function') {
        return;
    }

    try {
        if (typeof window.MathJax.typesetClear === 'function') {
            window.MathJax.typesetClear([element]);
        }
        window.MathJax.typesetPromise([element]).catch(error => {
            console.error('MathJax typeset error:', error);
        });
    } catch (error) {
        console.error('MathJax render error:', error);
    }
}


function renderAsiValue(value, noDistribution) {
    const container = document.getElementById("asi_value");
    if (!container) return;

    if (noDistribution) {
        container.innerHTML = '';
        return;
    }

    const numericValue = Number.parseFloat(value);
    const formattedValue = Number.isFinite(numericValue)
        ? numericValue.toFixed(4)
        : 'N/A';
    const asiExplanationHtml = String.raw`
        <div style="font-weight:600; margin-bottom:0.35rem;">Extreme-scenario ASI</div>
        <div style="margin-bottom:0.5rem;">
            The ASI is evaluated on the extreme-scenario matrix, i.e. the feasible solutions obtained by minimizing
            and maximizing each criterion weight in turn.
        </div>
        <div style="margin-bottom:0.45rem;">
            Let \(W = [w_{ij}] \in [0,100]^{m \times n}\), where \(w_{ij}\) is the weight of criterion \(j\)
            in extreme scenario \(i\), expressed in percent. Here \(m\) is the number of extreme scenarios and
            \(n\) is the number of criteria.
        </div>
        <div class="asi-math-block">
            \[
            \mathrm{ASI}(W)
            =
            1
            -
            \frac{1}{n}
            \sum_{j=1}^{n}
            \frac{
                \sqrt{
                    m \sum_{i=1}^{m} w_{ij}^{2}
                    -
                    \left(\sum_{i=1}^{m} w_{ij}\right)^{2}
                }
            }{
                100\left(\frac{m}{n}\right)\sqrt{n-1}
            }
            \]
        </div>
        <div style="margin-bottom:0.4rem;">
            Equivalently, if
            \( s_j = \sqrt{m \sum_{i=1}^{m} w_{ij}^{2} - \left(\sum_{i=1}^{m} w_{ij}\right)^{2}} \),
            then:
        </div>
        <div class="asi-math-block">
            \[
            \mathrm{ASI}(W)
            =
            1
            -
            \frac{1}{n}
            \sum_{j=1}^{n}
            \frac{s_j}{100(m/n)\sqrt{n-1}}
            \]
        </div>
        <div>
            Larger ASI values indicate lower dispersion of criterion weights across the extreme feasible cases,
            hence greater stability.
        </div>
    `;

    container.innerHTML = `
        <div class="asi-value-pill">
            <span class="asi-value-label">Average Stability Index (ASI)</span>
            <strong class="asi-value-number">${formattedValue}</strong>
            <span class="info-popover asi-info-popover">
                <button type="button" class="info-popover-trigger" aria-label="Explain the Average Stability Index">What is this?</button>
                <span class="info-popover-content">
                    ${asiExplanationHtml}
                </span>
            </span>
        </div>
    `;
    typesetMathInElement(container);
}


function renderInconsistencyReport(report) {
    const container = document.getElementById('results-container');
    if (!container) return;
    clearInconsistencyReport();

    const panel = document.createElement('div');
    panel.id = 'inconsistency-report';
    panel.style.border = '1px solid #e6b365';
    panel.style.borderRadius = '0.5rem';
    panel.style.padding = '0.75rem 0.9rem';
    panel.style.marginBottom = '0.8rem';
    panel.style.background = '#fffaf2';
    panel.style.fontSize = '0.9rem';
    panel.style.lineHeight = '1.45';

    const suggestions = Array.isArray(report?.suggestions) ? report.suggestions : [];
    const suggestionsHtml = suggestions.length
        ? suggestions.map(item => {
            const recommendations = Array.isArray(item.recommendations) ? item.recommendations : [];
            const recommendationLines = recommendations
                .map(rec => `<li>${rec.recommendation}</li>`)
                .join('');
            return `
                <div style="margin-top:0.55rem; padding-top:0.45rem; border-top:1px dashed #e8d8b9;">
                    <div style="font-weight:600;">Suggestion ${item.suggestion_id} (minimal changes: ${item.minimal_changes})</div>
                    <ul style="margin:0.25rem 0 0.1rem 1.2rem; padding:0;">
                        ${recommendationLines}
                    </ul>
                </div>
            `;
        }).join('')
        : `<div style="margin-top:0.45rem;">No actionable minimal suggestions were found.</div>`;

    panel.innerHTML = `
        <div style="font-weight:700; margin-bottom:0.2rem;">Inconsistency Identified</div>
        <div>${report?.message || 'The provided inputs are inconsistent.'}</div>
        <div style="margin-top:0.25rem; color:#5a6372;">
            Requested suggestions: ${report?.requested_suggestions ?? 3},
            Returned suggestions: ${report?.returned_suggestions ?? suggestions.length}
        </div>
        ${suggestionsHtml}
    `;

    container.prepend(panel);
}


document.querySelector('.calculate-button').addEventListener('click', () => {
    /*
    This function transfers all user inputs to backend, where calculations are done based on the
    revised Simos' method. Then, it reads back the results and displays the criteria weights.
    */

    // Before starting the calculation, remove tables with older results that may still be contained on the page
    document.getElementById("asi_value").textContent = '';
    document.querySelectorAll('#results-container table').forEach(table => table.remove());
    clearInconsistencyReport();
    Plotly.purge('boxplot');
    Plotly.purge('extreme_plot');
    Plotly.purge('pca_plot');
    if (typeof setFigureCardVisibility === 'function') {
        setFigureCardVisibility('boxplot-panel', false);
        setFigureCardVisibility('extreme-panel', false);
        setFigureCardVisibility('pca-panel', false);
    }

    // Declare variables for user inputs (arrangement of cards, z value, and required precision)
    const dropZone = document.querySelector('.drop-zone');
    const cards_arrangement = Array.from(dropZone.children).map((card) => {
        return {
            id: card.id,
            name: card.textContent.split('Delete')[0],
            class: card.classList.contains('criterion') ? 'criterion' : 'white',
            col: card.style.gridColumnStart
        };
    });

    let zValue;
    let eValue;
    let optionalConstraints = collectOptionalConstraintsFromDom();
    const inconsistencySuggestions = collectInconsistencySuggestionCountFromDom();
    const wValue = document.getElementById('w-value').value;
    const srf_method = document.getElementById('srf_method').value;

    const selectedMethod = document.getElementById("srf_method").value;
    const modularOptions = selectedMethod === 'modular_srf' && typeof collectModularOptionsFromDom === 'function'
        ? collectModularOptionsFromDom()
        : null;
    const samplingSize = typeof collectSamplingSizeFromDom === 'function'
        ? collectSamplingSizeFromDom()
        : null;
    // The dropdown value stays "modular_srf", but the payload also includes the
    // derived classical profile so both frontend and backend agree on input shape.
    const methodForInputs = selectedMethod === 'modular_srf' && typeof deriveModularInputProfile === 'function'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;

    const collectZBeliefOnly = () => {
        const rows = mergeBeliefRows(collectBeliefRowsFromDom('z'), false);
        const sum = rows.reduce((acc, row) => acc + row.beta, 0);
        if (!rows.length) throw new Error("Global z belief distribution requires at least one row.");
        if (rows.some(row => row.value <= 1 || row.beta <= 0)) throw new Error("Each z belief row must satisfy z > 1 and beta > 0.");
        if (Math.abs(sum - 1.0) > 1e-4) throw new Error("Global z belief betas must sum to 1.");
        const out = {};
        rows.forEach((row, idx) => {
            const i = idx + 1;
            out[`z-value-${i}`] = row.value;
            out[`z-beta-${i}`] = row.beta;
        });
        return out;
    };

    const collectEBeliefOnly = () => {
        const out = {};
        for (let rank = 1; rank < num_ranks; rank++) {
            const rows = mergeBeliefRows(collectBeliefRowsFromDom('e', rank), true);
            if (!rows.length) {
                throw new Error(`Gap rank ${rank}->${rank + 1} requires at least one belief row.`);
            }
            const sum = rows.reduce((acc, row) => acc + row.beta, 0);
            if (rows.some(row => !Number.isInteger(row.value) || row.value < 0 || row.beta <= 0)) {
                throw new Error(`Gap rank ${rank}->${rank + 1} requires e >= 0 integer and beta > 0.`);
            }
            if (Math.abs(sum - 1.0) > 1e-4) {
                throw new Error(`Gap rank ${rank}->${rank + 1} belief betas must sum to 1.`);
            }
            rows.forEach((row, idx) => {
                const i = idx + 1;
                out[`e-value-${rank}-${i}`] = row.value;
                out[`e-beta-${rank}-${i}`] = row.beta;
            });
        }
        return out;
    };

    const collectE0BeliefOnly = () => {
        const rows = mergeBeliefRows(collectBeliefRowsFromDom('e', 0), true);
        const sum = rows.reduce((acc, row) => acc + row.beta, 0);
        if (!rows.length) throw new Error("Zero-criterion e0 belief distribution requires at least one row.");
        if (rows.some(row => !Number.isInteger(row.value) || row.value < 0 || row.beta <= 0)) {
            throw new Error("Each e0 belief row must satisfy integer e0 >= 0 and beta > 0.");
        }
        if (Math.abs(sum - 1.0) > 1e-4) {
            throw new Error("Zero-criterion e0 belief betas must sum to 1.");
        }
        const out = {};
        rows.forEach((row, idx) => {
            const i = idx + 1;
            out[`e-value-0-${i}`] = row.value;
            out[`e-beta-0-${i}`] = row.beta;
        });
        return out;
    };

    if (selectedMethod === 'modular_srf' && modularOptions) {
        // Mirror the backend decision tree so we only serialize the inputs that are
        // active for the chosen modular questionnaire answers.
        const procedure = modularOptions.procedure || 'standard';
        const distanceType = modularOptions.distance_type || 'precise';
        const distanceFormat = modularOptions.distance_format || 'interval';
        const zType = procedure === 'standard' ? (modularOptions.z_type || 'precise') : 'na';
        const zFormat = modularOptions.z_format || 'interval';
        const distanceImprecise = distanceType === 'imprecise';
        const zImprecise = procedure === 'standard' && zType === 'imprecise';
        const allImpreciseAreInterval = (
            (!distanceImprecise || distanceFormat === 'interval')
            && (!zImprecise || zFormat === 'interval')
        );
        const probabilityAllowed = procedure !== 'direct'
            && (distanceImprecise || zImprecise)
            && allImpreciseAreInterval;
        const useProbability = modularOptions.probability === 'yes' && probabilityAllowed;

        if (procedure === 'direct') {
            zValue = {};
            if (distanceType === 'precise') {
                for (let i = 1; i < num_ranks; i++) {
                    const zExact = parseFloat(document.getElementById(`zexact-${i}`).value);
                    zValue[`zmin_${i}`] = zExact;
                    zValue[`zmax_${i}`] = zExact;
                }
            } else if (distanceFormat === 'fuzzy') {
                for (let i = 1; i < num_ranks; i++) {
                    zValue[`zmin_${i}`] = parseInt(document.getElementById(`wap-hfl-zmin-${i}`).value, 10);
                    zValue[`zmax_${i}`] = parseInt(document.getElementById(`wap-hfl-zmax-${i}`).value, 10);
                }
            } else {
                for (let i = 1; i < num_ranks; i++) {
                    zValue[`zmin_${i}`] = parseFloat(document.getElementById(`zmin-${i}`).value);
                    zValue[`zmax_${i}`] = parseFloat(document.getElementById(`zmax-${i}`).value);
                }
            }
            eValue = 0;
        } else if (procedure === 'zero') {
            zValue = 1;
            if (distanceType === 'imprecise') {
                if (useProbability) {
                    try {
                        eValue = {
                            ...collectEBeliefOnly(),
                            ...collectE0BeliefOnly()
                        };
                    } catch (error) {
                        alert(error.message || 'Invalid imprecise belief-degree inputs.');
                        return;
                    }
                } else if (distanceFormat === 'fuzzy') {
                    eValue = {};
                    eValue['rmin_0'] = parseInt(document.getElementById('e0-rmin-term').value, 10);
                    eValue['rmax_0'] = parseInt(document.getElementById('e0-rmax-term').value, 10);
                    for (let i = 1; i < num_ranks; i++) {
                        eValue[`rmin_${i}`] = parseInt(document.getElementById(`hfl-rmin-${i}`).value, 10);
                        eValue[`rmax_${i}`] = parseInt(document.getElementById(`hfl-rmax-${i}`).value, 10);
                    }
                } else {
                    eValue = {};
                    eValue['emin_0'] = parseFloat(document.getElementById('e0min-value').value);
                    eValue['emax_0'] = parseFloat(document.getElementById('e0max-value').value);
                    for (let i = 1; i < num_ranks; i++) {
                        eValue[`emin_${i}`] = parseFloat(document.getElementById(`emin-${i}`).value);
                        eValue[`emax_${i}`] = parseFloat(document.getElementById(`emax-${i}`).value);
                    }
                }
            } else {
                const e0Value = parseInt(document.getElementById('e0-value')?.value, 10);
                eValue = Number.isInteger(e0Value) ? e0Value : 0;
            }
        } else {
            if (useProbability) {
                try {
                    zValue = collectZBeliefOnly();
                } catch (error) {
                    alert(error.message || 'Invalid z belief-degree inputs.');
                    return;
                }
            } else if (zType === 'imprecise') {
                if (zFormat === 'fuzzy') {
                    zValue = {
                        emin: parseInt(document.getElementById('hfl-emin-term').value, 10),
                        emax: parseInt(document.getElementById('hfl-emax-term').value, 10)
                    };
                } else {
                    zValue = {
                        zmin: parseFloat(document.getElementById('zmin').value),
                        zmax: parseFloat(document.getElementById('zmax').value)
                    };
                }
            } else {
                zValue = document.getElementById('z-value').value;
            }

            if (distanceType === 'imprecise') {
                if (useProbability) {
                    try {
                        eValue = collectEBeliefOnly();
                    } catch (error) {
                        alert(error.message || 'Invalid gap belief-degree inputs.');
                        return;
                    }
                } else if (distanceFormat === 'fuzzy') {
                    eValue = {};
                    for (let i = 1; i < num_ranks; i++) {
                        eValue[`rmin_${i}`] = parseInt(document.getElementById(`hfl-rmin-${i}`).value, 10);
                        eValue[`rmax_${i}`] = parseInt(document.getElementById(`hfl-rmax-${i}`).value, 10);
                    }
                } else {
                    eValue = {};
                    for (let i = 1; i < num_ranks; i++) {
                        eValue[`emin_${i}`] = parseFloat(document.getElementById(`emin-${i}`).value);
                        eValue[`emax_${i}`] = parseFloat(document.getElementById(`emax-${i}`).value);
                    }
                }
            } else {
                eValue = 0;
            }
        }
    } else {
        // Format the z values based on the selected method
        if (methodForInputs === 'srf_ii') {
            // Zero-criterion variant does not require z input.
            zValue = 1;
        } else if (methodForInputs === 'wap') {
            zValue = {}
            for (let i = 1; i < num_ranks; i++) {
                zValue[`zmin_${i}`] = parseFloat(document.getElementById(`zmin-${i}`).value);
                zValue[`zmax_${i}`] = parseFloat(document.getElementById(`zmax-${i}`).value);
            }
        } else if (methodForInputs === 'imprecise_srf') {
            zValue = {
                zmin: parseFloat(document.getElementById('zmin').value),
                zmax: parseFloat(document.getElementById('zmax').value)
            };
        } else if (methodForInputs === 'belief_degree_imprecise_srf') {
            try {
                const serializedBeliefInputs = serializeBeliefInputsFromDom();
                zValue = serializedBeliefInputs.zValue;
                eValue = serializedBeliefInputs.eValue;
            } catch (error) {
                alert(error.message || 'Invalid belief-degree inputs. Please review the distributions.');
                return;
            }
        } else if (methodForInputs === 'hfl_srf') {
            zValue = {
                emin: parseInt(document.getElementById('hfl-emin-term').value, 10),
                emax: parseInt(document.getElementById('hfl-emax-term').value, 10)
            };

            if (zValue.emin < 1 || zValue.emin > 10 || zValue.emax < 1 || zValue.emax > 10) {
                alert("For HFL-SRF, global z terms must be within 1 to 10.");
                return;
            }
            if (zValue.emin > zValue.emax) {
                alert("For HFL-SRF, lower z-term must be less than or equal to upper z-term. Please recheck your inputs.");
                return;
            }
        } else {
            zValue = document.getElementById('z-value').value;
        }

        // Format the e values based on the selected method
        if (methodForInputs === 'imprecise_srf') {
            eValue = {}
            for (let i = 1; i < num_ranks; i++) {
                const eMinEl = document.getElementById(`emin-${i}`);
                const eMaxEl = document.getElementById(`emax-${i}`);
                if (!eMinEl || !eMaxEl) {
                    alert(`Missing imprecise gap inputs for rank pair ${i}-${i + 1}. Please refresh the method panel and try again.`);
                    return;
                }
                eValue[`emin_${i}`] = parseFloat(eMinEl.value);
                eValue[`emax_${i}`] = parseFloat(eMaxEl.value);
            }
        } else if (methodForInputs === 'belief_degree_imprecise_srf') {
            // already serialized in z branch for belief-degree inputs
        } else if (methodForInputs === 'hfl_srf') {
            eValue = {};
            for (let i = 1; i < num_ranks; i++) {
                const rMin = parseInt(document.getElementById(`hfl-rmin-${i}`).value, 10);
                const rMax = parseInt(document.getElementById(`hfl-rmax-${i}`).value, 10);

                if (rMin < 1 || rMin > 5 || rMax < 1 || rMax > 5) {
                    alert(`For HFL-SRF, rank-gap terms must be within 1 to 5 for rank pair ${i}-${i + 1}.`);
                    return;
                }
                if (rMin > rMax) {
                    alert(`For HFL-SRF, lower rank-gap term must be less than or equal to upper rank-gap term for rank pair ${i}-${i + 1}.`);
                    return;
                }

                eValue[`rmin_${i}`] = rMin;
                eValue[`rmax_${i}`] = rMax;
            }
        } else {
            // Only SRF-II needs e0 from the DOM; crisp SRF/Robust/WAP use e=0.
            if (methodForInputs === 'srf_ii') {
                const e0Value = parseInt(document.getElementById('e0-value')?.value, 10);
                eValue = Number.isInteger(e0Value) ? e0Value : 0;
            } else {
                eValue = 0;
            }
        }
    }

    // Define spinner options
    const opts = {
        lines: 15, // The number of lines to draw
        length: 30, // The length of each line
        width: 30, // The line thickness
        radius: 84, // The radius of the inner circle
        scale: 0.2, // Scales overall size of the spinner
        corners: 1, // Corner roundness (0..1)
        speed: 1.2, // Rounds per second
        rotate: 0, // The rotation offset
        animation: 'spinner-line-fade-quick', // The CSS animation name for the lines
        direction: 1, // 1: clockwise, -1: counterclockwise
        color: '#9f9f9f', // CSS color or array of colors
        fadeColor: 'transparent', // CSS color or array of colors
        top: '50%', // Top position relative to parent
        left: '50%', // Left position relative to parent
        zIndex: 2000000000, // The z-index (defaults to 2e9)
        className: 'spinner', // The CSS class to assign to the spinner
        position: 'absolute', // Element positioning
    };

    // Create the spinner
    const target = document.getElementById('loader-container');
    const spinner = new Spinner(opts);

    const missingZ = zValue === null || zValue === undefined || zValue === '';
    const missingE = eValue === null || eValue === undefined || eValue === '';
    const missingW = wValue === null || wValue === undefined || wValue === '';
    if (missingZ || missingE || missingW) {
        alert('Please make sure to provide all input values.');
    } else {
        // Start spinner
        target.style.display = 'block';
        spinner.spin(target);
        startCalculationProgressPolling();

        fetch('/calculate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                cards_arrangement,
                zValue,
                eValue,
                wValue,
                srf_method,
                optionalConstraints,
                inconsistencySuggestions,
                samplingSize,
                modularOptions,
                // The server still re-resolves modularOptions; this profile is a
                // parsing hint that keeps import/export flows straightforward.
                modularProfile: methodForInputs
            })
        })
            .then(async response => {
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    const message = payload?.error || `Calculation failed (HTTP ${response.status}).`;
                    throw new Error(message);
                }
                return payload;
            })
            .then(data => {
                if (data?.inconsistency?.detected) {
                    renderInconsistencyReport(data.inconsistency);
                    stopCalculationProgressPolling();
                    spinner.stop();
                    target.style.display = 'none';
                    return;
                }

                // Display the calculation results to the user in a tabular form
                simos_calc_results = JSON.parse(data.crit_weights);
                asi_value = data.asi_value;

                const distributionMethods = new Set([
                    'robust_srf',
                    'wap',
                    'imprecise_srf',
                    'belief_degree_imprecise_srf',
                    'hfl_srf'
                ]);
                const modularNoDistribution = selectedMethod === 'modular_srf'
                    ? ((modularOptions?.output_type || 'single') !== 'variability')
                    : null;
                const noDistribution = selectedMethod === 'modular_srf'
                    ? modularNoDistribution
                    : !distributionMethods.has(selectedMethod);
                // Reuse the classical display rules for modular tables so the UI can
                // hide/show center columns without duplicating rendering logic.
                const tableMethod = selectedMethod === 'modular_srf'
                    ? (noDistribution ? 'srf' : 'robust_srf')
                    : selectedMethod;
                renderAsiValue(asi_value, noDistribution);
                createTableFromDataframe(simos_calc_results, tableMethod);
                plot_boxplot(simos_calc_results, noDistribution, container_id = 'boxplot');
                plot_extreme_scenarios(noDistribution, container_id = 'extreme_plot');
                plot_pca(noDistribution, container_id = 'pca_plot')

                // Stop spinner
                stopCalculationProgressPolling();
                spinner.stop();
                target.style.display = 'none';
            })
            .catch(error => {
                console.error('Error:', error);
                alert(error.message || 'Calculation failed. Please review your inputs and try again.');

                // stop spinner
                stopCalculationProgressPolling();
                spinner.stop();
                target.style.display = 'none';
            });
    }
});


function updateSimosMethodInfo(selected) {
    /*
    This function updates the block with information related to the selected extension of the revised Simos method.
    */

    const e0_block = document.getElementById('e0_value_query');
    const z_block = document.getElementById('z_value_query');

    if (selected === 'srf_ii') {
        e0_block.style.display = 'flex';
        z_block.style.display = 'none';
    } else if (selected === 'imprecise_srf') {
        e0_block.style.display = 'flex';
        z_block.style.display = 'flex';
    } else if (selected === 'belief_degree_imprecise_srf') {
        e0_block.style.display = 'flex';
        z_block.style.display = 'flex';
    } else if (selected === 'hfl_srf') {
        e0_block.style.display = 'flex';
        z_block.style.display = 'flex';
    } else if (selected === 'modular_srf') {
        e0_block.style.display = 'flex';
        z_block.style.display = 'flex';
    } else {
        e0_block.style.display = 'none';
        z_block.style.display = 'flex';
    }
    updateGridState()

    // Modify the instructions div based on the selected SRF method
    fetch('/static/data/simos_instructions.json', { cache: 'no-store' })
        .then(response => response.json())
        .then(async data => {
            document.getElementById("method-full-name").textContent = data[selected].full_name;
            document.getElementById("method-short-name").textContent = data[selected].short_name;
            document.getElementById("methodology").innerHTML = data[selected].methodology;

            // Keep methodology visible by default on each method switch.
            const guidelinesButton = document.getElementById("tool-user-guidelines");
            const guidelinesContent = guidelinesButton ? guidelinesButton.nextElementSibling : null;
            if (guidelinesButton && guidelinesContent) {
                guidelinesButton.classList.add("active");
                requestAnimationFrame(() => {
                    guidelinesContent.style.maxHeight = guidelinesContent.scrollHeight + "px";
                });
            }
            document.getElementById("doi-link").textContent = 'loading...';

            const doi = encodeURIComponent(data[selected].doi_link);
            const response = await fetch(`https://citation.doi.org/format?doi=${doi}&style=apa`, {
                headers: {"Accept": "text/x-bibliography"}
            });

            let doi_fullref = await response.text();
            doi_fullref = doi_fullref.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
            document.getElementById("doi-link").innerHTML = doi_fullref;
        })
        .catch(error => console.error("Error loading content:", error));
}
