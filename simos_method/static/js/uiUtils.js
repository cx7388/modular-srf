let num_ranks = 0;
let white_cards = {};
const HFL_Z_TERM_MIN = 1;
const HFL_Z_TERM_MAX = 10;
const HFL_CARD_TERM_MIN = 1;
const HFL_CARD_TERM_MAX = 5;

const HFL_Z_TERM_DEFS = [
    { value: 1, label: 'very low contrast', fuzzy: [1, 1, 2] },
    { value: 2, label: 'low contrast', fuzzy: [1, 2, 3] },
    { value: 3, label: 'slightly low contrast', fuzzy: [2, 3, 4] },
    { value: 4, label: 'moderate-low contrast', fuzzy: [3, 4, 5] },
    { value: 5, label: 'medium contrast', fuzzy: [4, 5, 6] },
    { value: 6, label: 'moderate-high contrast', fuzzy: [5, 6, 7] },
    { value: 7, label: 'high contrast', fuzzy: [6, 7, 8] },
    { value: 8, label: 'very high contrast', fuzzy: [7, 8, 9] },
    { value: 9, label: 'extremely high contrast', fuzzy: [8, 9, 10] },
    { value: 10, label: 'maximum contrast', fuzzy: [9, 10, 10] }
];

const HFL_CARD_TERM_DEFS = [
    { value: 1, label: 'very small gap', fuzzy: [1, 1, 2] },
    { value: 2, label: 'small gap', fuzzy: [1, 2, 3] },
    { value: 3, label: 'medium gap', fuzzy: [2, 3, 4] },
    { value: 4, label: 'large gap', fuzzy: [3, 4, 5] },
    { value: 5, label: 'very large gap', fuzzy: [4, 5, 5] }
];

const METHOD_INPUT_MEMORY = {};


function rememberMethodInputs(root = document) {
    if (!root || typeof root.querySelectorAll !== 'function') return;
    root.querySelectorAll('input[id], select[id], textarea[id]').forEach(el => {
        if (!el.id) return;
        if (el.type === 'checkbox') {
            METHOD_INPUT_MEMORY[el.id] = Boolean(el.checked);
        } else {
            METHOD_INPUT_MEMORY[el.id] = el.value;
        }
    });
}


function restoreMethodInputs(root = document) {
    if (!root || typeof root.querySelectorAll !== 'function') return;
    root.querySelectorAll('input[id], select[id], textarea[id]').forEach(el => {
        if (!el.id || !(el.id in METHOD_INPUT_MEMORY)) return;
        if (el.type === 'checkbox') {
            el.checked = Boolean(METHOD_INPUT_MEMORY[el.id]);
        } else {
            el.value = METHOD_INPUT_MEMORY[el.id];
        }
    });
}


function getRememberedBeliefMaxIndex(scope, rank = null) {
    const keys = Object.keys(METHOD_INPUT_MEMORY);
    let maxIdx = 0;

    if (scope === 'z') {
        keys.forEach(key => {
            const match = /^z-(value|beta)-(\d+)$/.exec(key);
            if (match) maxIdx = Math.max(maxIdx, parseInt(match[2], 10));
        });
    } else {
        const escapedRank = String(rank);
        const regex = new RegExp(`^e-(value|beta)-${escapedRank}-(\\d+)$`);
        keys.forEach(key => {
            const match = regex.exec(key);
            if (match) maxIdx = Math.max(maxIdx, parseInt(match[2], 10));
        });
    }

    return maxIdx;
}


function ensureRememberedBeliefRows(scope, rank = null) {
    if (typeof addBeliefRow !== 'function') return;

    const maxIdx = getRememberedBeliefMaxIndex(scope, rank);
    if (maxIdx <= 1) return;

    if (scope === 'z') {
        for (let idx = 2; idx <= maxIdx; idx++) {
            if (!document.getElementById(`z-value-${idx}`)) {
                addBeliefRow('z');
            }
        }
        return;
    }

    if (!document.getElementById(`e-value-${rank}-1`)) return;
    for (let idx = 2; idx <= maxIdx; idx++) {
        if (!document.getElementById(`e-value-${rank}-${idx}`)) {
            addBeliefRow('e', rank);
            if (!document.getElementById(`e-value-${rank}-${idx}`)) break;
        }
    }
}


function buildHflTermOptions(termDefinitions, selectedValue) {
    return termDefinitions
        .map(term => {
            const selected = term.value === selectedValue ? 'selected' : '';
            const fuzzy = `(${term.fuzzy.join(',')})`;
            return `<option value="${term.value}" ${selected}>${term.value} - ${term.label} | ${fuzzy}</option>`;
        })
        .join('');
}


function buildHflDefinitionsTable(termDefinitions, title) {
    const rowsHtml = termDefinitions
        .map(term => `<tr><td>${term.value}</td><td>${term.label}</td><td>(${term.fuzzy.join(',')})</td></tr>`)
        .join('');

    return `
        <div style="flex:1; min-width: 18rem;">
            <div style="font-weight:600; margin-bottom:0.2rem;">${title}</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.78rem;">
                <thead>
                    <tr>
                        <th style="text-align:left; border-bottom:1px solid #d8dfe8; padding:0.12rem 0.2rem;">Term</th>
                        <th style="text-align:left; border-bottom:1px solid #d8dfe8; padding:0.12rem 0.2rem;">Meaning</th>
                        <th style="text-align:left; border-bottom:1px solid #d8dfe8; padding:0.12rem 0.2rem;">Fuzzy set</th>
                    </tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}


function buildHflDefinitionsHtml() {
    return `
        <div style="display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem; border:1px dashed #d3d9e2; border-radius:0.45rem; padding:0.5rem;">
            ${buildHflDefinitionsTable(HFL_Z_TERM_DEFS, 'Global contrast terms (z: 1 to 10)')}
            ${buildHflDefinitionsTable(HFL_CARD_TERM_DEFS, 'Rank-gap terms (cards: 1 to 5)')}
        </div>
    `;
}


function buildOptionalConstraintsHtml() {
    return `
        <div id="optional-constraints-panel" style="margin-top:0.8rem; border:1px dashed #d3d9e2; border-radius:0.45rem; padding:0.55rem;">
            <div style="font-weight:600; margin-bottom:0.35rem;">Optional Requirements (Off by Default)</div>
            <div style="font-size:0.78rem; color:#6f7784; margin-bottom:0.45rem;">
                Enable only if needed. Feasibility is checked automatically when calculating.
            </div>

            <div style="display:flex; gap:0.4rem; flex-wrap:wrap; align-items:center; margin-bottom:0.45rem;">
                <label for="opt-inconsistency-suggestions" style="min-width:15rem;">Inconsistency suggestions to return</label>
                <input type="number" id="opt-inconsistency-suggestions" class="labelmaxmin form-control"
                       step="1" min="1" max="20" value="3" oninput="enforceMinMaxLimits(event)">
                <span style="font-size:0.75rem; color:#6f7784;">default = 3</span>
            </div>

            <label style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">
                <input type="checkbox" id="opt-enable-dictatorship">
                Apply anti-dictatorship requirement
            </label>
            <div id="opt-dictatorship-panel" style="display:none; margin:0.25rem 0 0.5rem 1.25rem; padding-left:0.55rem; border-left:2px solid #d3d9e2;">
                <div style="font-size:0.78rem; color:#6f7784;">
                    Automatic rule: each criterion weight must not exceed the sum of all other criteria weights.
                </div>
            </div>

            <label style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">
                <input type="checkbox" id="opt-enable-minweight">
                Apply minimum-weight requirement (all criteria)
            </label>
            <div id="opt-minweight-panel" style="display:none; margin:0.25rem 0 0.1rem 1.25rem; padding-left:0.55rem; border-left:2px solid #d3d9e2;">
                <div style="display:flex; gap:0.4rem; flex-wrap:wrap; align-items:center;">
                    <label for="opt-minweight-value" style="min-width:9rem;">Minimum weight</label>
                    <input type="number" id="opt-minweight-value" class="labelmaxmin form-control"
                           step="0.1" min="0" max="100" value="0" oninput="enforceMinMaxLimits(event)">
                    <span style="font-size:0.75rem; color:#6f7784;">% for each criterion</span>
                </div>
            </div>
        </div>
    `;
}


function syncOptionalConstraintPanels() {
    const dictatorshipPanel = document.getElementById('opt-dictatorship-panel');
    const minWeightPanel = document.getElementById('opt-minweight-panel');
    const dictatorshipEnabled = document.getElementById('opt-enable-dictatorship')?.checked;
    const minWeightEnabled = document.getElementById('opt-enable-minweight')?.checked;

    if (dictatorshipPanel) {
        dictatorshipPanel.style.display = dictatorshipEnabled ? 'block' : 'none';
    }
    if (minWeightPanel) {
        minWeightPanel.style.display = minWeightEnabled ? 'block' : 'none';
    }
}


function ensureOptionalConstraintsPanel() {
    const addInputsContainer = document.getElementById("additional-inputs");
    if (!addInputsContainer) return;

    let panel = document.getElementById('optional-constraints-panel');
    if (!panel) {
        const holder = document.createElement('div');
        holder.innerHTML = buildOptionalConstraintsHtml().trim();
        panel = holder.firstElementChild;
        const summaryPanel = document.getElementById("method-summary-panel");
        addInputsContainer.insertBefore(panel, summaryPanel || null);
    }
    syncOptionalConstraintPanels();
}


function getModularDefaultOptions() {
    return {
        procedure: 'standard',          // standard | zero | direct
        distance_type: 'precise',      // precise | imprecise
        distance_format: 'interval',   // interval | fuzzy
        z_type: 'precise',             // precise | imprecise
        z_format: 'interval',          // interval | fuzzy
        probability: 'no',             // no | yes
        output_type: 'single',         // single | variability
        unit_weight: 'fixed',          // fixed | dynamic
        variability_method: 'sampling' // sampling + extreme scenarios
    };
}


function collectModularOptionsFromDom() {
    // Keep these defaults and coercion rules aligned with the backend
    // MODULAR_DEFAULT_OPTIONS/_normalize_modular_options path.
    const defaults = getModularDefaultOptions();
    const valueOrDefault = (id, fallback) => {
        const el = document.getElementById(id);
        return el && typeof el.value === 'string' && el.value !== '' ? el.value : fallback;
    };

    const options = {
        procedure: valueOrDefault('mod-q3-procedure', defaults.procedure),
        distance_type: valueOrDefault('mod-q4-distance', defaults.distance_type),
        distance_format: valueOrDefault('mod-q5-distance-format', defaults.distance_format),
        z_type: valueOrDefault('mod-q6-z', defaults.z_type),
        z_format: valueOrDefault('mod-q7-z-format', defaults.z_format),
        probability: valueOrDefault('mod-q8-prob', defaults.probability),
        output_type: valueOrDefault('mod-q11-output', defaults.output_type),
        unit_weight: valueOrDefault('mod-q12-unit', defaults.unit_weight),
        variability_method: valueOrDefault('mod-q13-var-method', defaults.variability_method)
    };

    if (options.procedure === 'zero') {
        options.z_type = 'na';
    }
    if (options.procedure === 'direct') {
        options.z_type = 'na';
        options.probability = 'no';
        options.unit_weight = 'fixed';
    }
    if (options.procedure === 'standard' && options.z_type === 'na') {
        options.z_type = 'precise';
    }

    const distanceImprecise = options.distance_type === 'imprecise';
    const zImprecise = options.procedure === 'standard' && options.z_type === 'imprecise';
    const allImpreciseAreInterval = (
        (!distanceImprecise || options.distance_format === 'interval')
        && (!zImprecise || options.z_format === 'interval')
    );
    const probabilityAllowed = options.procedure !== 'direct'
        && (distanceImprecise || zImprecise)
        && allImpreciseAreInterval;
    if (!probabilityAllowed) {
        options.probability = 'no';
    }

    // If probability is requested for standard procedure, z is also modeled probabilistically.
    if (options.procedure === 'standard' && options.probability === 'yes') {
        options.z_type = 'imprecise';
        options.z_format = 'interval';
    }
    return options;
}


const MAX_SAMPLING_INPUT_SIZE = 20000;


function getCriterionCardCountFromDom() {
    return Math.max(1, document.querySelectorAll('.drop-zone .card.criterion').length);
}


function usesSamplingForCurrentConfiguration(selectedMethod = null, modularOptions = null, methodForInputs = null) {
    const method = selectedMethod || document.getElementById("srf_method")?.value;
    const options = modularOptions || (method === 'modular_srf' ? collectModularOptionsFromDom() : null);
    const profile = methodForInputs || (method === 'modular_srf'
        ? deriveModularInputProfile(options)
        : method);

    if (method === 'modular_srf') {
        return Boolean(
            options
            && options.output_type === 'variability'
        );
    }

    return new Set([
        'robust_srf',
        'wap',
        'imprecise_srf',
        'belief_degree_imprecise_srf',
        'hfl_srf'
    ]).has(profile);
}


function getDefaultSamplingSizeForConfiguration(selectedMethod = null, modularOptions = null, methodForInputs = null) {
    const method = selectedMethod || document.getElementById("srf_method")?.value;
    const options = modularOptions || (method === 'modular_srf' ? collectModularOptionsFromDom() : null);
    const profile = methodForInputs || (method === 'modular_srf'
        ? deriveModularInputProfile(options)
        : method);

    if (method === 'modular_srf' && options) {
        return options.output_type === 'variability' ? 200 : null;
    }

    return new Set([
        'robust_srf',
        'wap',
        'imprecise_srf',
        'belief_degree_imprecise_srf',
        'hfl_srf'
    ]).has(profile) ? 200 : null;
}


function ensureSamplingSizePanel() {
    const container = document.getElementById("additional-inputs");
    if (!container) return null;

    let panel = document.getElementById('sampling-size-query');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'sampling-size-query';
        panel.style.display = 'none';
        panel.style.justifyContent = 'space-between';
        panel.style.alignItems = 'center';
        panel.style.gap = '1rem';
        panel.style.marginTop = '0.6rem';
        panel.innerHTML = `
            <label for="sampling-size-input">
                How many feasible samples should be generated for the variability analysis?
                <div style="font-size:0.78rem; color:#6f7784; margin-top:0.2rem;">
                    Dynamic analysis always includes both sampling and extreme scenarios. This value controls the sampling run only.
                </div>
            </label>
            <input type="number" id="sampling-size-input" name="sampling-size-input"
                   class="labelmaxmin form-control"
                   min="1" max="${MAX_SAMPLING_INPUT_SIZE}" step="1" value="200"
                   placeholder="Samples">
        `;

        const input = panel.querySelector('#sampling-size-input');
        input?.addEventListener('input', () => {
            input.dataset.userEdited = 'true';
            validateMethodInputsAndToggleRun();
        });

        const wQuery = document.getElementById('w_value_query');
        if (wQuery?.parentNode) {
            wQuery.insertAdjacentElement('afterend', panel);
        } else {
            container.appendChild(panel);
        }
    }

    return panel;
}


function updateSamplingSizeVisibility() {
    const selectedMethod = document.getElementById("srf_method")?.value;
    const modularOptions = selectedMethod === 'modular_srf' ? collectModularOptionsFromDom() : null;
    const methodForInputs = selectedMethod === 'modular_srf'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;
    const panel = ensureSamplingSizePanel();
    const input = document.getElementById('sampling-size-input');
    if (!panel || !input) return;

    const shouldShow = usesSamplingForCurrentConfiguration(selectedMethod, modularOptions, methodForInputs);
    panel.style.display = shouldShow ? 'flex' : 'none';
    if (!shouldShow) {
        return;
    }

    const defaultValue = getDefaultSamplingSizeForConfiguration(selectedMethod, modularOptions, methodForInputs);
    const modeKey = [
        selectedMethod || '',
        methodForInputs || '',
        modularOptions?.output_type || ''
    ].join('|');

    if (input.dataset.modeKey !== modeKey) {
        input.value = String(defaultValue ?? 200);
        input.dataset.modeKey = modeKey;
        input.dataset.userEdited = 'false';
        return;
    }

    if (!input.value || Number.parseInt(input.value, 10) <= 0) {
        input.value = String(defaultValue ?? 200);
    }
}


function collectSamplingSizeFromDom() {
    const panel = document.getElementById('sampling-size-query');
    const input = document.getElementById('sampling-size-input');
    if (!panel || !input || panel.style.display === 'none') {
        return null;
    }

    const parsedValue = parseInt(input.value, 10);
    if (!Number.isInteger(parsedValue) || parsedValue <= 0) {
        return null;
    }
    return Math.min(parsedValue, MAX_SAMPLING_INPUT_SIZE);
}


function validateSamplingSizeInput() {
    const panel = document.getElementById('sampling-size-query');
    const input = document.getElementById('sampling-size-input');
    if (!panel || !input || panel.style.display === 'none') {
        return {
            valid: true,
            summaryHtml: ''
        };
    }

    const parsedValue = parseInt(input.value, 10);
    const valid = Number.isInteger(parsedValue)
        && parsedValue >= 1
        && parsedValue <= MAX_SAMPLING_INPUT_SIZE;

    const summaryHtml = valid
        ? `<div style="margin:0.35rem 0 0.15rem;">Sampling count: ${parsedValue} feasible solutions requested.</div>`
        : `<div style="margin:0.35rem 0 0.15rem; color:#b54708;">Sampling count must be an integer between 1 and ${MAX_SAMPLING_INPUT_SIZE}.</div>`;

    return {valid, summaryHtml};
}


function deriveModularInputProfile(options = null) {
    const cfg = options || collectModularOptionsFromDom();

    // This is a UI-facing approximation used to choose which controls to render.
    // The backend resolves the final profile again before solving.
    if (cfg.procedure === 'direct') {
        return 'wap';
    }
    if (cfg.procedure === 'zero') {
        return 'srf_ii';
    }

    // standard procedure: map combinations to implemented SRF variants
    const hasImpreciseDistance = cfg.distance_type === 'imprecise';
    const hasImpreciseZ = cfg.z_type === 'imprecise';

    if ((hasImpreciseDistance || hasImpreciseZ) && cfg.probability === 'yes') {
        return 'belief_degree_imprecise_srf';
    }
    if ((hasImpreciseDistance && cfg.distance_format === 'fuzzy')
        || (hasImpreciseZ && cfg.z_format === 'fuzzy')) {
        return 'hfl_srf';
    }
    if (hasImpreciseDistance || hasImpreciseZ) {
        return 'imprecise_srf';
    }
    return 'srf';
}

function isWhiteCardLockedMode(selectedMethod = null, modularOptions = null, methodForInputs = null) {
    const method = selectedMethod || document.getElementById("srf_method")?.value;
    const options = modularOptions || (method === 'modular_srf' ? collectModularOptionsFromDom() : null);
    const profile = methodForInputs || (method === 'modular_srf'
        ? deriveModularInputProfile(options)
        : method);

    if (method === 'modular_srf' && options) {
        const modularImpreciseDistance = (options.procedure === 'standard' || options.procedure === 'zero')
            && options.distance_type === 'imprecise';
        return options.procedure === 'direct' || modularImpreciseDistance;
    }

    const noBlankMethods = new Set(['wap', 'hfl_srf', 'imprecise_srf', 'belief_degree_imprecise_srf']);
    return noBlankMethods.has(profile);
}


function updateModularQuestionnaireVisibility() {
    const selectedMethod = document.getElementById("srf_method")?.value;
    const panel = document.getElementById('modular-config-query');
    if (!panel) return;

    const isModular = selectedMethod === 'modular_srf';
    panel.style.display = isModular ? 'block' : 'none';
    if (!isModular) return;

    const options = collectModularOptionsFromDom();
    const procedure = options.procedure;

    const rowQ4 = document.getElementById('mod-row-q4');
    const rowQ5 = document.getElementById('mod-row-q5');
    const rowQ6 = document.getElementById('mod-row-q6');
    const rowQ7 = document.getElementById('mod-row-q7');
    const rowQ8 = document.getElementById('mod-row-q8');
    const rowQ12 = document.getElementById('mod-row-q12');
    const rowQ13 = document.getElementById('mod-row-q13');

    const standardProcedure = procedure === 'standard';
    const directProcedure = procedure === 'direct';
    const showDistanceType = true;
    const showDistanceFormat = options.distance_type === 'imprecise';
    const showZType = standardProcedure;
    const showZFormat = standardProcedure && options.z_type === 'imprecise' && options.probability !== 'yes';
    const distanceImprecise = options.distance_type === 'imprecise';
    const zImprecise = standardProcedure && options.z_type === 'imprecise';
    const allImpreciseAreInterval = (
        (!distanceImprecise || options.distance_format === 'interval')
        && (!zImprecise || options.z_format === 'interval')
    );
    const showProbability = !directProcedure
        && (distanceImprecise || zImprecise)
        && allImpreciseAreInterval;
    const showUnitWeight = !directProcedure;
    const showVariabilityMethod = options.output_type === 'variability';

    if (rowQ4) rowQ4.style.display = showDistanceType ? 'flex' : 'none';
    if (rowQ5) rowQ5.style.display = showDistanceFormat ? 'flex' : 'none';
    if (rowQ6) rowQ6.style.display = showZType ? 'flex' : 'none';
    if (rowQ7) rowQ7.style.display = showZFormat ? 'flex' : 'none';
    if (rowQ8) rowQ8.style.display = showProbability ? 'flex' : 'none';
    if (rowQ12) rowQ12.style.display = showUnitWeight ? 'flex' : 'none';
    if (rowQ13) rowQ13.style.display = showVariabilityMethod ? 'flex' : 'none';

    if (!showDistanceFormat) {
        const q5 = document.getElementById('mod-q5-distance-format');
        if (q5) q5.value = 'interval';
    }
    if (!showZFormat) {
        const q7 = document.getElementById('mod-q7-z-format');
        if (q7) q7.value = 'interval';
    }
    if (!showProbability) {
        const q8 = document.getElementById('mod-q8-prob');
        if (q8) q8.value = 'no';
    }
    if (!showUnitWeight) {
        const q12 = document.getElementById('mod-q12-unit');
        if (q12) q12.value = 'fixed';
    }
    if (!showVariabilityMethod) {
        const q13 = document.getElementById('mod-q13-var-method');
        if (q13) q13.value = 'sampling';
    }

    const zTypeSelect = document.getElementById('mod-q6-z');
    if (zTypeSelect) {
        zTypeSelect.disabled = !standardProcedure;
        if (!standardProcedure) {
            zTypeSelect.value = 'precise';
        } else if (zTypeSelect.value === 'na') {
            zTypeSelect.value = 'precise';
        }
    }

    if (!standardProcedure) {
        const q7 = document.getElementById('mod-q7-z-format');
        if (q7) q7.value = 'interval';
    }

    if (procedure === 'zero') {
        const q6 = document.getElementById('mod-q6-z');
        const q7 = document.getElementById('mod-q7-z-format');
        if (q6) q6.value = 'precise';
        if (q7) q7.value = 'interval';
    }

    if (standardProcedure && options.probability === 'yes') {
        const q6 = document.getElementById('mod-q6-z');
        const q7 = document.getElementById('mod-q7-z-format');
        if (q6) q6.value = 'imprecise';
        if (q7) q7.value = 'interval';
    }
}


function ensureModularQuestionnairePanel() {
    const container = document.getElementById("additional-inputs");
    if (!container) return;

    const selectedMethod = document.getElementById("srf_method")?.value;
    let panel = document.getElementById('modular-config-query');
    if (!panel) {
        const defaults = getModularDefaultOptions();
        panel = document.createElement('div');
        panel.id = 'modular-config-query';
        panel.style.border = '1px dashed #d3d9e2';
        panel.style.borderRadius = '0.45rem';
        panel.style.padding = '0.55rem';
        panel.style.marginBottom = '0.75rem';
        panel.innerHTML = `
            <div style="font-weight:600; margin-bottom:0.35rem;">Modular SRF Questionnaire</div>
            <div style="font-size:0.78rem; color:#6f7784; margin-bottom:0.45rem;">
                Select components to assemble your Modular SRF variant.
            </div>

            <div style="display:flex; flex-direction:column; gap:0.35rem;">
                <div style="display:flex; gap:0.45rem; align-items:center;">
                    <label for="mod-q3-procedure" style="min-width:18rem;">Procedural mechanism</label>
                    <select id="mod-q3-procedure" class="labelmaxmin form-control">
                        <option value="standard" selected>Standard deck (blank cards + global z)</option>
                        <option value="zero">Zero-criterion (SRF-II)</option>
                        <option value="direct">Direct-ratio (WAP style)</option>
                    </select>
                </div>

                <div id="mod-row-q4" style="display:flex; gap:0.45rem; align-items:center;">
                    <label for="mod-q4-distance" style="min-width:18rem;">Distance input type</label>
                    <select id="mod-q4-distance" class="labelmaxmin form-control">
                        <option value="precise" selected>Precise</option>
                        <option value="imprecise">Imprecise</option>
                    </select>
                </div>

                <div id="mod-row-q5" style="display:none; gap:0.45rem; align-items:center;">
                    <label for="mod-q5-distance-format" style="min-width:18rem;">Distance imprecision format</label>
                    <select id="mod-q5-distance-format" class="labelmaxmin form-control">
                        <option value="interval" selected>Interval</option>
                        <option value="fuzzy">Fuzzy (HFL)</option>
                    </select>
                </div>

                <div id="mod-row-q6" style="display:flex; gap:0.45rem; align-items:center;">
                    <label for="mod-q6-z" style="min-width:18rem;">Global ratio z input type</label>
                    <select id="mod-q6-z" class="labelmaxmin form-control">
                        <option value="precise" selected>Precise</option>
                        <option value="imprecise">Imprecise</option>
                    </select>
                </div>

                <div id="mod-row-q7" style="display:none; gap:0.45rem; align-items:center;">
                    <label for="mod-q7-z-format" style="min-width:18rem;">Global ratio z imprecision format</label>
                    <select id="mod-q7-z-format" class="labelmaxmin form-control">
                        <option value="interval" selected>Interval</option>
                        <option value="fuzzy">Fuzzy (HFL)</option>
                    </select>
                </div>

                <div id="mod-row-q8" style="display:none; gap:0.45rem; align-items:center;">
                    <label for="mod-q8-prob" style="min-width:18rem;">Probability for imprecise inputs</label>
                    <select id="mod-q8-prob" class="labelmaxmin form-control">
                        <option value="no" selected>No</option>
                        <option value="yes">Yes</option>
                    </select>
                </div>

                <div style="display:flex; gap:0.45rem; align-items:center;">
                    <label for="mod-q11-output" style="min-width:18rem;">Output type</label>
                    <select id="mod-q11-output" class="labelmaxmin form-control">
                        <option value="single" selected>Single weight vector</option>
                        <option value="variability">Variability analysis</option>
                    </select>
                </div>

                <div id="mod-row-q12" style="display:flex; gap:0.45rem; align-items:center;">
                    <label for="mod-q12-unit" style="min-width:18rem;">Unit weight C</label>
                    <select id="mod-q12-unit" class="labelmaxmin form-control">
                        <option value="fixed" selected>Fixed</option>
                        <option value="dynamic">Dynamic</option>
                    </select>
                </div>

                <div id="mod-row-q13" style="display:none; gap:0.45rem; align-items:center;">
                    <label for="mod-q13-var-method" style="min-width:18rem;">Dynamic analysis outputs</label>
                    <select id="mod-q13-var-method" class="labelmaxmin form-control" disabled>
                        <option value="sampling" selected>Sampling + extreme scenarios</option>
                    </select>
                    <span style="font-size:0.75rem; color:#6f7784;">Both outputs are always generated together.</span>
                </div>
            </div>
        `;

        const zBlock = document.getElementById("z_value_query");
        container.insertBefore(panel, zBlock || container.firstChild);

        panel.querySelectorAll('select').forEach(selectEl => {
            selectEl.addEventListener('change', () => updateGridState());
        });

        // ensure defaults are set explicitly for import/export consistency
        document.getElementById('mod-q3-procedure').value = defaults.procedure;
        document.getElementById('mod-q4-distance').value = defaults.distance_type;
        document.getElementById('mod-q5-distance-format').value = defaults.distance_format;
        document.getElementById('mod-q6-z').value = defaults.z_type;
        document.getElementById('mod-q7-z-format').value = defaults.z_format;
        document.getElementById('mod-q8-prob').value = defaults.probability;
        document.getElementById('mod-q11-output').value = defaults.output_type;
        document.getElementById('mod-q12-unit').value = defaults.unit_weight;
        document.getElementById('mod-q13-var-method').value = defaults.variability_method;
    }

    panel.style.display = selectedMethod === 'modular_srf' ? 'block' : 'none';
    updateModularQuestionnaireVisibility();
}


const BELIEF_BETA_TOLERANCE = 1e-4;


function ensureMethodSummaryPanel() {
    const container = document.getElementById("additional-inputs");
    if (!container) return null;

    let panel = document.getElementById("method-summary-panel");
    if (!panel) {
        panel = document.createElement("div");
        panel.id = "method-summary-panel";
        Object.assign(panel.style, {
            marginTop: "0.8rem",
            border: "1px solid #d9dee5",
            borderRadius: "0.5rem",
            padding: "0.65rem 0.8rem",
            background: "#fbfcfe",
            fontSize: "0.88rem",
            lineHeight: "1.45"
        });
        container.appendChild(panel);
    }
    return panel;
}


function setMethodSummary(html, isWarning = false) {
    const panel = ensureMethodSummaryPanel();
    if (!panel) return;
    panel.style.display = 'block';
    panel.style.borderColor = isWarning ? '#e6b365' : '#d9dee5';
    panel.innerHTML = html;
}


function hideMethodSummary() {
    const panel = document.getElementById("method-summary-panel");
    if (panel) {
        panel.style.display = 'none';
    }
}


function buildBeliefRowHtml(scope, index, value, beta, rank = null, locked = false) {
    const valueId = scope === 'z' ? `z-value-${index}` : `e-value-${rank}-${index}`;
    const betaId = scope === 'z' ? `z-beta-${index}` : `e-beta-${rank}-${index}`;
    const rowId = scope === 'z' ? `z-row-${index}` : `e-row-${rank}-${index}`;
    const valueStep = scope === 'z' ? '0.1' : '1';
    const valueMin = scope === 'z' ? '1.01' : '0';
    const valuePlaceholder = scope === 'z' ? 'z (ratio)' : 'e (# blank cards)';
    const deleteBtn = locked
        ? ''
        : `<button type="button" title="Delete row" style="padding: 0.2rem 0.45rem; border: 1px solid #d8dde5; border-radius: 0.32rem; background: #fff; cursor: pointer;"
                   onclick="deleteBeliefRow('${scope}', ${rank === null ? 'null' : rank}, ${index})"><i class="fa-solid fa-trash-can"></i></button>`;
    const readonlyAttr = locked ? 'readonly' : '';
    const disabledAttr = locked ? 'disabled' : '';

    return `
        <div id="${rowId}" style="display: grid; grid-template-columns: 1fr 1fr auto; gap: 0.4rem; margin-bottom: 0.25rem;">
            <input type="number" id="${valueId}" class="labelmaxmin form-control"
                   step="${valueStep}" min="${valueMin}" max="1000" value="${value}" ${readonlyAttr}
                   placeholder="${valuePlaceholder}"
                   oninput="enforceMinMaxLimits(event); validateMethodInputsAndToggleRun();">
            <input type="number" id="${betaId}" class="labelmaxmin form-control"
                   step="0.05" min="0" max="1" value="${beta}" ${disabledAttr}
                   placeholder="beta"
                   oninput="enforceMinMaxLimits(event); validateMethodInputsAndToggleRun();">
            ${deleteBtn}
        </div>
    `;
}


function getBeliefRows(scope, rank = null) {
    const prefix = scope === 'z' ? 'z-value-' : `e-value-${rank}-`;
    const valueInputs = Array.from(document.querySelectorAll(`input[id^='${prefix}']`));
    const rows = [];

    valueInputs.forEach(input => {
        const suffix = scope === 'z'
            ? input.id.split('z-value-')[1]
            : input.id.split(`e-value-${rank}-`)[1];
        const betaId = scope === 'z' ? `z-beta-${suffix}` : `e-beta-${rank}-${suffix}`;
        const betaInput = document.getElementById(betaId);
        if (!betaInput) return;

        rows.push({
            index: parseInt(suffix, 10),
            value: parseFloat(input.value),
            beta: parseFloat(betaInput.value),
            valueInput: input,
            betaInput
        });
    });

    rows.sort((a, b) => a.index - b.index);
    return rows;
}


function getNextBeliefRowIndex(scope, rank = null) {
    const rows = getBeliefRows(scope, rank);
    if (rows.length === 0) return 1;
    return Math.max(...rows.map(row => row.index)) + 1;
}


function addBeliefRow(scope, rank = null) {
    const containerId = scope === 'z' ? 'belief-z-rows' : `belief-gap-${rank}-rows`;
    const container = document.getElementById(containerId);
    if (!container) return;

    const idx = getNextBeliefRowIndex(scope, rank);
    const defaultValue = scope === 'z'
        ? 6.5
        : (rank === 0
            ? 4
            : Math.max(0, parseInt(String(white_cards[rank] ?? 0), 10)));
    container.insertAdjacentHTML(
        'beforeend',
        buildBeliefRowHtml(scope, idx, defaultValue, 0.5, rank, false)
    );
    validateMethodInputsAndToggleRun();
}


function deleteBeliefRow(scope, rank, index) {
    const rowId = scope === 'z' ? `z-row-${index}` : `e-row-${rank}-${index}`;
    document.getElementById(rowId)?.remove();
    validateMethodInputsAndToggleRun();
}


function normalizeBeliefRows(scope, rank = null) {
    const rows = getBeliefRows(scope, rank);
    const sum = rows.reduce((acc, row) => acc + (Number.isFinite(row.beta) ? row.beta : 0), 0);
    if (sum <= 0) return;

    rows.forEach(row => {
        row.betaInput.value = (row.beta / sum).toFixed(4);
    });
    validateMethodInputsAndToggleRun();
}


function clearBeliefRows(scope, rank = null) {
    const containerId = scope === 'z' ? 'belief-z-rows' : `belief-gap-${rank}-rows`;
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    addBeliefRow(scope, rank);
}


function useObservedGapBelief(rank) {
    const container = document.getElementById(`belief-gap-${rank}-rows`);
    if (!container) return;
    const observedGap = Math.max(0, parseInt(String(white_cards[rank] ?? 0), 10));
    container.innerHTML = buildBeliefRowHtml('e', 1, observedGap, 1, rank, false);
    validateMethodInputsAndToggleRun();
}


function isNearOne(sum) {
    return Math.abs(sum - 1) <= BELIEF_BETA_TOLERANCE;
}


function updateBeliefSumIndicator(indicatorId, sum, valid) {
    const indicator = document.getElementById(indicatorId);
    if (!indicator) return;

    indicator.textContent = `sum(beta) = ${sum.toFixed(4)}`;
    indicator.style.fontWeight = '500';
    indicator.style.color = valid ? '#2f7d32' : '#b54708';
}


function collectBeliefRowsMerged(scope, rank = null) {
    const rows = getBeliefRows(scope, rank)
        .filter(row => Number.isFinite(row.value) && Number.isFinite(row.beta))
        .map(row => ({
            value: scope === 'z' ? row.value : Math.round(row.value),
            beta: row.beta
        }));

    const merged = new Map();
    rows.forEach(row => {
        const key = row.value;
        merged.set(key, (merged.get(key) || 0) + row.beta);
    });

    return Array.from(merged.entries())
        .map(([value, beta]) => ({ value: Number(value), beta: Number(beta) }))
        .sort((a, b) => a.value - b.value);
}


function validateBeliefMethodInputs() {
    const errors = [];
    const summaryLines = [];

    const zRows = collectBeliefRowsMerged('z');
    const zSum = zRows.reduce((acc, row) => acc + row.beta, 0);
    const zSumValid = isNearOne(zSum);
    const zRowsValid = zRows.length > 0 && zRows.every(row => row.value > 1 && row.beta > 0);
    updateBeliefSumIndicator('belief-z-sum', zSum, zSumValid && zRowsValid);

    if (!zRowsValid) {
        errors.push('Global z distribution requires z>1 and beta>0 for all rows.');
    }
    if (!zSumValid) {
        errors.push('Global z betas must sum to 1.');
    }
    if (zRows.length > 0) {
        summaryLines.push(`<li><strong>z:</strong> support [${zRows[0].value}, ${zRows[zRows.length - 1].value}], sum(beta)=${zSum.toFixed(4)}</li>`);
    }

    for (let rank = 1; rank < num_ranks; rank++) {
        const gapRows = collectBeliefRowsMerged('e', rank);
        const gapSum = gapRows.reduce((acc, row) => acc + row.beta, 0);
        const gapSumValid = isNearOne(gapSum);
        const gapRowsValid = gapRows.length > 0 && gapRows.every(row => Number.isInteger(row.value) && row.value >= 0 && row.beta > 0);
        updateBeliefSumIndicator(`belief-gap-${rank}-sum`, gapSum, gapSumValid && gapRowsValid);

        if (!gapRowsValid) {
            errors.push(`Gap Rank ${rank}->${rank + 1} requires e>=0 integer and beta>0.`);
        }
        if (!gapSumValid) {
            errors.push(`Gap Rank ${rank}->${rank + 1} betas must sum to 1.`);
        }
        if (gapRows.length > 0) {
            summaryLines.push(`<li><strong>Gap ${rank}->${rank + 1}:</strong> support [${gapRows[0].value}, ${gapRows[gapRows.length - 1].value}], sum(beta)=${gapSum.toFixed(4)}</li>`);
        }
    }

    const summaryHtml = `
        <div style="font-weight: 600; margin-bottom: 0.25rem;">Belief-Degree Summary</div>
        <ul style="margin: 0.1rem 0 0.25rem 1rem; padding: 0;">
            ${summaryLines.join('')}
        </ul>
        ${errors.length ? `<div style="color:#b54708;">${errors.join('<br>')}</div>` : '<div style="color:#2f7d32;">All belief distributions are valid.</div>'}
    `;

    return { valid: errors.length === 0, summaryHtml };
}


function validateModularMethodInputs(modularOptions) {
    const errors = [];
    const summaryLines = [];
    const opts = modularOptions || getModularDefaultOptions();
    const isStandard = opts.procedure === 'standard';
    const isZero = opts.procedure === 'zero';
    const isDirect = opts.procedure === 'direct';
    const distanceImprecise = opts.distance_type === 'imprecise';
    const zImprecise = isStandard && opts.z_type === 'imprecise';
    const allImpreciseAreInterval = (
        (!distanceImprecise || opts.distance_format === 'interval')
        && (!zImprecise || opts.z_format === 'interval')
    );
    const probabilityAllowed = !isDirect
        && (distanceImprecise || zImprecise)
        && allImpreciseAreInterval;
    const useProb = opts.probability === 'yes' && probabilityAllowed;
    const useProbDistance = useProb && opts.distance_type === 'imprecise';
    const useProbZ = useProb && isStandard;

    if (opts.probability === 'yes' && !probabilityAllowed) {
        errors.push('Probability can be enabled only when imprecision format is interval.');
    }

    if (isDirect) {
        summaryLines.push('<li><strong>Procedure:</strong> direct-ratio (WAP)</li>');
        if (num_ranks < 2) {
            errors.push('WAP requires at least two ranks.');
        }
        if (opts.distance_type === 'precise') {
            summaryLines.push('<li><strong>Successive ratios:</strong> precise</li>');
            for (let i = 1; i < num_ranks; i++) {
                const zExact = parseFloat(document.getElementById(`zexact-${i}`)?.value);
                if (!Number.isFinite(zExact)) {
                    errors.push(`Rank pair ${i}-${i + 1} requires a valid exact ratio value.`);
                    continue;
                }
                if (zExact <= 1) {
                    errors.push(`Rank pair ${i}-${i + 1} requires ratio > 1.`);
                }
            }
        } else if (opts.distance_format === 'fuzzy') {
            summaryLines.push('<li><strong>Successive ratios:</strong> imprecise fuzzy terms</li>');
            for (let i = 1; i < num_ranks; i++) {
                const zMinTerm = parseInt(document.getElementById(`wap-hfl-zmin-${i}`)?.value, 10);
                const zMaxTerm = parseInt(document.getElementById(`wap-hfl-zmax-${i}`)?.value, 10);
                if (!Number.isInteger(zMinTerm) || zMinTerm < HFL_Z_TERM_MIN || zMinTerm > HFL_Z_TERM_MAX) {
                    errors.push(`Rank pair ${i}-${i + 1} lower fuzzy term must be in [${HFL_Z_TERM_MIN}, ${HFL_Z_TERM_MAX}].`);
                }
                if (!Number.isInteger(zMaxTerm) || zMaxTerm < HFL_Z_TERM_MIN || zMaxTerm > HFL_Z_TERM_MAX) {
                    errors.push(`Rank pair ${i}-${i + 1} upper fuzzy term must be in [${HFL_Z_TERM_MIN}, ${HFL_Z_TERM_MAX}].`);
                }
                if (Number.isInteger(zMinTerm) && Number.isInteger(zMaxTerm) && zMinTerm > zMaxTerm) {
                    errors.push(`Rank pair ${i}-${i + 1} requires lower fuzzy term <= upper fuzzy term.`);
                }
            }
        } else {
            summaryLines.push('<li><strong>Successive ratios:</strong> imprecise interval</li>');
            for (let i = 1; i < num_ranks; i++) {
                const zMin = parseFloat(document.getElementById(`zmin-${i}`)?.value);
                const zMax = parseFloat(document.getElementById(`zmax-${i}`)?.value);
                if (!Number.isFinite(zMin) || !Number.isFinite(zMax)) {
                    errors.push(`Rank pair ${i}-${i + 1} requires valid z_min and z_max.`);
                    continue;
                }
                if (zMin <= 1 || zMax <= 1) {
                    errors.push(`Rank pair ${i}-${i + 1} requires z values > 1.`);
                }
                if (zMin > zMax) {
                    errors.push(`Rank pair ${i}-${i + 1} requires z_min <= z_max.`);
                }
            }
        }
    } else if (isZero) {
        summaryLines.push('<li><strong>Procedure:</strong> zero-criterion (SRF-II)</li>');
        if (opts.distance_type === 'precise') {
            const e0 = parseInt(document.getElementById('e0-value')?.value, 10);
            if (!Number.isInteger(e0) || e0 < 0) {
                errors.push('e0 must be an integer >= 0.');
            }
        }

        if (opts.distance_type === 'imprecise' && !useProbDistance && opts.distance_format === 'interval') {
            const e0Min = parseFloat(document.getElementById('e0min-value')?.value);
            const e0Max = parseFloat(document.getElementById('e0max-value')?.value);
            if (!Number.isFinite(e0Min) || !Number.isFinite(e0Max)) {
                errors.push('e0 interval requires valid min and max values.');
            } else {
                if (!Number.isInteger(e0Min) || !Number.isInteger(e0Max)) {
                    errors.push('e0 interval bounds must be integers.');
                }
                if (e0Min < 0 || e0Max < 0) errors.push('e0 interval bounds must be >= 0.');
                if (e0Min > e0Max) errors.push('e0 interval requires min <= max.');
            }

            for (let rank = 1; rank < num_ranks; rank++) {
                const eMin = parseFloat(document.getElementById(`emin-${rank}`)?.value);
                const eMax = parseFloat(document.getElementById(`emax-${rank}`)?.value);
                if (!Number.isFinite(eMin) || !Number.isFinite(eMax)) {
                    errors.push(`Gap ${rank}->${rank + 1} requires valid interval bounds.`);
                    continue;
                }
                if (eMin < 0 || eMax < 0) errors.push(`Gap ${rank}->${rank + 1} interval must be >= 0.`);
                if (eMin > eMax) errors.push(`Gap ${rank}->${rank + 1} requires min <= max.`);
            }
        } else if (opts.distance_type === 'imprecise' && !useProbDistance && opts.distance_format === 'fuzzy') {
            const e0MinTerm = parseInt(document.getElementById('e0-rmin-term')?.value, 10);
            const e0MaxTerm = parseInt(document.getElementById('e0-rmax-term')?.value, 10);
            if (!Number.isInteger(e0MinTerm) || e0MinTerm < HFL_CARD_TERM_MIN || e0MinTerm > HFL_CARD_TERM_MAX) {
                errors.push(`e0 lower fuzzy term must be in [${HFL_CARD_TERM_MIN}, ${HFL_CARD_TERM_MAX}].`);
            }
            if (!Number.isInteger(e0MaxTerm) || e0MaxTerm < HFL_CARD_TERM_MIN || e0MaxTerm > HFL_CARD_TERM_MAX) {
                errors.push(`e0 upper fuzzy term must be in [${HFL_CARD_TERM_MIN}, ${HFL_CARD_TERM_MAX}].`);
            }
            if (Number.isInteger(e0MinTerm) && Number.isInteger(e0MaxTerm) && e0MinTerm > e0MaxTerm) {
                errors.push('e0 fuzzy interval must satisfy smaller gap <= larger gap.');
            }

            const hflState = validateHflMethodInputs();
            if (!hflState.valid) {
                errors.push('HFL rank-gap intervals are invalid.');
            }
        } else if (opts.distance_type === 'imprecise' && useProbDistance) {
            const e0Rows = collectBeliefRowsMerged('e', 0);
            const e0Sum = e0Rows.reduce((acc, row) => acc + row.beta, 0);
            const e0RowsValid = e0Rows.length > 0
                && e0Rows.every(row => Number.isInteger(row.value) && row.value >= 0 && row.beta > 0);
            const e0SumValid = isNearOne(e0Sum);
            updateBeliefSumIndicator('belief-gap-0-sum', e0Sum, e0RowsValid && e0SumValid);
            if (!e0RowsValid) errors.push('e0 belief rows require integer e0>=0 and beta>0.');
            if (!e0SumValid) errors.push('e0 belief betas must sum to 1.');

            for (let rank = 1; rank < num_ranks; rank++) {
                const gapRows = collectBeliefRowsMerged('e', rank);
                const gapSum = gapRows.reduce((acc, row) => acc + row.beta, 0);
                const gapRowsValid = gapRows.length > 0 && gapRows.every(row => Number.isInteger(row.value) && row.value >= 0 && row.beta > 0);
                const gapSumValid = isNearOne(gapSum);
                updateBeliefSumIndicator(`belief-gap-${rank}-sum`, gapSum, gapRowsValid && gapSumValid);
                if (!gapRowsValid) errors.push(`Gap ${rank}->${rank + 1} requires e>=0 integer and beta>0.`);
                if (!gapSumValid) errors.push(`Gap ${rank}->${rank + 1} belief betas must sum to 1.`);
            }
        }
    } else {
        summaryLines.push('<li><strong>Procedure:</strong> standard deck</li>');
        if (opts.z_type === 'na') {
            errors.push('In standard deck, global z input type cannot be "Not used".');
        }
        if (useProb && opts.z_type !== 'imprecise') {
            errors.push('When probability is enabled, global z must be modeled as a belief distribution.');
        }

        if (opts.z_type === 'precise') {
            const z = parseFloat(document.getElementById('z-value')?.value);
            if (!Number.isFinite(z) || z <= 1) {
                errors.push('Global z must be a number > 1.');
            }
        } else if (opts.z_type === 'imprecise' && !useProbZ && opts.z_format === 'interval') {
            const zMin = parseFloat(document.getElementById('zmin')?.value);
            const zMax = parseFloat(document.getElementById('zmax')?.value);
            if (!Number.isFinite(zMin) || !Number.isFinite(zMax)) {
                errors.push('Global z interval requires valid min and max values.');
            } else {
                if (zMin <= 1 || zMax <= 1) errors.push('Global z interval bounds must be > 1.');
                if (zMin > zMax) errors.push('Global z interval requires min <= max.');
            }
        } else if (opts.z_type === 'imprecise' && !useProbZ && opts.z_format === 'fuzzy') {
            const zState = validateHflMethodInputs();
            if (!zState.valid) {
                errors.push('HFL global z interval is invalid.');
            }
        } else if (opts.z_type === 'imprecise' && useProbZ) {
            const zRows = collectBeliefRowsMerged('z');
            const zSum = zRows.reduce((acc, row) => acc + row.beta, 0);
            const zRowsValid = zRows.length > 0 && zRows.every(row => row.value > 1 && row.beta > 0);
            const zSumValid = isNearOne(zSum);
            updateBeliefSumIndicator('belief-z-sum', zSum, zRowsValid && zSumValid);
            if (!zRowsValid) errors.push('Global z belief rows require z>1 and beta>0.');
            if (!zSumValid) errors.push('Global z belief betas must sum to 1.');
        }

        if (opts.distance_type === 'imprecise' && !useProbDistance && opts.distance_format === 'interval') {
            for (let rank = 1; rank < num_ranks; rank++) {
                const eMin = parseFloat(document.getElementById(`emin-${rank}`)?.value);
                const eMax = parseFloat(document.getElementById(`emax-${rank}`)?.value);
                if (!Number.isFinite(eMin) || !Number.isFinite(eMax)) {
                    errors.push(`Gap ${rank}->${rank + 1} requires valid interval bounds.`);
                    continue;
                }
                if (eMin < 0 || eMax < 0) errors.push(`Gap ${rank}->${rank + 1} interval must be >= 0.`);
                if (eMin > eMax) errors.push(`Gap ${rank}->${rank + 1} requires min <= max.`);
            }
        } else if (opts.distance_type === 'imprecise' && !useProbDistance && opts.distance_format === 'fuzzy') {
            const hflState = validateHflMethodInputs();
            if (!hflState.valid) {
                errors.push('HFL rank-gap intervals are invalid.');
            }
        } else if (opts.distance_type === 'imprecise' && useProbDistance) {
            for (let rank = 1; rank < num_ranks; rank++) {
                const gapRows = collectBeliefRowsMerged('e', rank);
                const gapSum = gapRows.reduce((acc, row) => acc + row.beta, 0);
                const gapRowsValid = gapRows.length > 0 && gapRows.every(row => Number.isInteger(row.value) && row.value >= 0 && row.beta > 0);
                const gapSumValid = isNearOne(gapSum);
                updateBeliefSumIndicator(`belief-gap-${rank}-sum`, gapSum, gapRowsValid && gapSumValid);
                if (!gapRowsValid) errors.push(`Gap ${rank}->${rank + 1} requires e>=0 integer and beta>0.`);
                if (!gapSumValid) errors.push(`Gap ${rank}->${rank + 1} belief betas must sum to 1.`);
            }
        }
    }

    summaryLines.push(`<li><strong>Distance input:</strong> ${opts.distance_type}</li>`);
    summaryLines.push(`<li><strong>z input:</strong> ${opts.z_type}</li>`);
    summaryLines.push(`<li><strong>Probability:</strong> ${opts.probability}</li>`);
    summaryLines.push(`<li><strong>Output type:</strong> ${opts.output_type}</li>`);
    if (!isDirect) {
        summaryLines.push(`<li><strong>Unit weight:</strong> ${opts.unit_weight}</li>`);
    }

    const summaryHtml = `
        <div style="font-weight: 600; margin-bottom: 0.25rem;">Modular Configuration Summary</div>
        <ul style="margin: 0.1rem 0 0.25rem 1rem; padding: 0;">
            ${summaryLines.join('')}
        </ul>
        ${errors.length ? `<div style="color:#b54708;">${errors.join('<br>')}</div>` : '<div style="color:#2f7d32;">Modular inputs are valid.</div>'}
    `;

    return { valid: errors.length === 0, summaryHtml };
}


function validateHflMethodInputs() {
    const errors = [];
    const summaryLines = [];

    const eminInput = document.getElementById('hfl-emin-term');
    const emaxInput = document.getElementById('hfl-emax-term');
    if (eminInput && emaxInput) {
        const emin = parseInt(eminInput.value, 10);
        const emax = parseInt(emaxInput.value, 10);
        if (!Number.isInteger(emin) || emin < HFL_Z_TERM_MIN || emin > HFL_Z_TERM_MAX) {
            errors.push(`Global lower contrast term must be in [${HFL_Z_TERM_MIN}, ${HFL_Z_TERM_MAX}].`);
        }
        if (!Number.isInteger(emax) || emax < HFL_Z_TERM_MIN || emax > HFL_Z_TERM_MAX) {
            errors.push(`Global upper contrast term must be in [${HFL_Z_TERM_MIN}, ${HFL_Z_TERM_MAX}].`);
        }
        if (emin > emax) {
            errors.push('Global importance contrast must satisfy lower <= upper.');
        }
        summaryLines.push(`<li><strong>Global contrast:</strong> [${emin}, ${emax}]</li>`);
    }

    for (let rank = 1; rank < num_ranks; rank++) {
        const rMinInput = document.getElementById(`hfl-rmin-${rank}`);
        const rMaxInput = document.getElementById(`hfl-rmax-${rank}`);
        if (!rMinInput || !rMaxInput) continue;
        const rMin = parseInt(rMinInput.value, 10);
        const rMax = parseInt(rMaxInput.value, 10);
        if (!Number.isInteger(rMin) || rMin < HFL_CARD_TERM_MIN || rMin > HFL_CARD_TERM_MAX) {
            errors.push(`Gap Rank ${rank}->${rank + 1} lower term must be in [${HFL_CARD_TERM_MIN}, ${HFL_CARD_TERM_MAX}].`);
        }
        if (!Number.isInteger(rMax) || rMax < HFL_CARD_TERM_MIN || rMax > HFL_CARD_TERM_MAX) {
            errors.push(`Gap Rank ${rank}->${rank + 1} upper term must be in [${HFL_CARD_TERM_MIN}, ${HFL_CARD_TERM_MAX}].`);
        }
        if (rMin > rMax) {
            errors.push(`Gap Rank ${rank}->${rank + 1} must satisfy smaller gap <= larger gap.`);
        }
        summaryLines.push(`<li><strong>Gap ${rank}->${rank + 1}:</strong> [${rMin}, ${rMax}]</li>`);
    }

    const summaryHtml = `
        <div style="font-weight: 600; margin-bottom: 0.25rem;">HFL-SRF Summary</div>
        <ul style="margin: 0.1rem 0 0.25rem 1rem; padding: 0;">
            ${summaryLines.join('')}
        </ul>
        ${errors.length ? `<div style="color:#b54708;">${errors.join('<br>')}</div>` : '<div style="color:#2f7d32;">All HFL inputs are valid.</div>'}
    `;

    return { valid: errors.length === 0, summaryHtml };
}


function validateOptionalConstraintsInputs() {
    const errors = [];
    const summaryLines = [];

    const dictatorshipEnabled = Boolean(document.getElementById('opt-enable-dictatorship')?.checked);
    const minWeightEnabled = Boolean(document.getElementById('opt-enable-minweight')?.checked);
    const nCriteria = document.querySelectorAll('.drop-zone .card.criterion').length;
    const suggestionCount = parseInt(document.getElementById('opt-inconsistency-suggestions')?.value, 10);

    if (!Number.isInteger(suggestionCount) || suggestionCount < 1 || suggestionCount > 20) {
        errors.push('Number of inconsistency suggestions must be an integer in [1, 20].');
    } else {
        summaryLines.push(`<li><strong>Inconsistency suggestions:</strong> ${suggestionCount}</li>`);
    }

    if (dictatorshipEnabled) {
        if (nCriteria < 2) {
            errors.push('Anti-dictatorship requires at least two criteria.');
        }
        summaryLines.push('<li><strong>Anti-dictatorship:</strong> enabled (w_i <= sum of all other weights)</li>');
    } else {
        summaryLines.push('<li><strong>Anti-dictatorship:</strong> not applied</li>');
    }

    if (minWeightEnabled) {
        const minWeightValue = parseFloat(document.getElementById('opt-minweight-value')?.value);
        if (!Number.isFinite(minWeightValue) || minWeightValue < 0 || minWeightValue > 100) {
            errors.push('Minimum-weight requirement must be within [0, 100].');
        } else if (nCriteria > 0 && minWeightValue * nCriteria > 100 + 1e-6) {
            errors.push('Minimum-weight requirement is infeasible: lower bounds exceed total weight 100.');
        }
        summaryLines.push('<li><strong>Minimum weight:</strong> enabled</li>');
    } else {
        summaryLines.push('<li><strong>Minimum weight:</strong> not applied</li>');
    }

    const summaryHtml = `
        <div style="font-weight: 600; margin: 0.35rem 0 0.25rem;">Optional Requirements</div>
        <ul style="margin: 0.1rem 0 0.25rem 1rem; padding: 0;">
            ${summaryLines.join('')}
        </ul>
        ${errors.length ? `<div style="color:#b54708;">${errors.join('<br>')}</div>` : '<div style="color:#2f7d32;">Optional requirements are valid.</div>'}
    `;

    return {
        valid: errors.length === 0,
        anyEnabled: dictatorshipEnabled || minWeightEnabled,
        summaryHtml
    };
}


function validateMethodInputsAndToggleRun() {
    const selectedMethod = document.getElementById("srf_method")?.value;
    const modularOptions = selectedMethod === 'modular_srf' ? collectModularOptionsFromDom() : null;
    const methodForInputs = selectedMethod === 'modular_srf'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;
    const runButton = document.querySelector('.calculate-button');
    if (!runButton) return;
    syncOptionalConstraintPanels();
    updateModularQuestionnaireVisibility();
    updateSamplingSizeVisibility();
    const optionalState = validateOptionalConstraintsInputs();
    const samplingState = validateSamplingSizeInput();

    if (selectedMethod === 'modular_srf') {
        const modularState = validateModularMethodInputs(modularOptions);
        const isValid = modularState.valid && optionalState.valid && samplingState.valid;
        const modularSummary = `
            <div style="font-size:0.8rem; color:#6f7784; margin-bottom:0.35rem;">
                Procedure=${modularOptions?.procedure || '-'}, distance=${modularOptions?.distance_type || '-'},
                z=${modularOptions?.z_type || '-'}, output=${modularOptions?.output_type || '-'}.
            </div>
        `;
        setMethodSummary(`${modularSummary}${modularState.summaryHtml}${samplingState.summaryHtml}${optionalState.summaryHtml}`, !isValid);
        runButton.disabled = !isValid;
        runButton.style.opacity = isValid ? '1' : '0.6';
        return;
    }

    if (methodForInputs === 'belief_degree_imprecise_srf') {
        const state = validateBeliefMethodInputs();
        const isValid = state.valid && optionalState.valid && samplingState.valid;
        const modularSummary = selectedMethod === 'modular_srf'
            ? `<div style="font-weight:600; margin-bottom:0.25rem;">Modular profile: ${methodForInputs}</div>`
            : '';
        setMethodSummary(`${modularSummary}${state.summaryHtml}${samplingState.summaryHtml}${optionalState.summaryHtml}`, !isValid);
        runButton.disabled = !isValid;
        runButton.style.opacity = isValid ? '1' : '0.6';
        return;
    }

    if (methodForInputs === 'hfl_srf') {
        const state = validateHflMethodInputs();
        const isValid = state.valid && optionalState.valid && samplingState.valid;
        const modularSummary = selectedMethod === 'modular_srf'
            ? `<div style="font-weight:600; margin-bottom:0.25rem;">Modular profile: ${methodForInputs}</div>`
            : '';
        setMethodSummary(`${modularSummary}${state.summaryHtml}${samplingState.summaryHtml}${optionalState.summaryHtml}`, !isValid);
        runButton.disabled = !isValid;
        runButton.style.opacity = isValid ? '1' : '0.6';
        return;
    }

    if (selectedMethod === 'modular_srf' || optionalState.anyEnabled || !optionalState.valid || samplingState.summaryHtml) {
        const modularSummary = selectedMethod === 'modular_srf'
            ? `
                <div style="font-weight:600; margin-bottom:0.25rem;">Modular profile: ${methodForInputs}</div>
                <div style="font-size:0.8rem; color:#6f7784; margin-bottom:0.35rem;">
                    Procedure=${modularOptions?.procedure || '-'}, distance=${modularOptions?.distance_type || '-'},
                    z=${modularOptions?.z_type || '-'}, output=${modularOptions?.output_type || '-'}.
                </div>
            `
            : '';
        setMethodSummary(`${modularSummary}${samplingState.summaryHtml}${optionalState.summaryHtml}`, !(optionalState.valid && samplingState.valid));
    } else {
        hideMethodSummary();
    }
    runButton.disabled = !(optionalState.valid && samplingState.valid);
    runButton.style.opacity = optionalState.valid && samplingState.valid ? '1' : '0.6';
}


function enforceMethodSpecificCardRules() {
    const selectedMethod = document.getElementById("srf_method")?.value;
    const modularOptions = selectedMethod === 'modular_srf' ? collectModularOptionsFromDom() : null;
    const methodForInputs = selectedMethod === 'modular_srf'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;
    let noBlankCards = isWhiteCardLockedMode(selectedMethod, modularOptions, methodForInputs);
    const modularImpreciseDistance = selectedMethod === 'modular_srf'
        && modularOptions
        && (modularOptions.procedure === 'standard' || modularOptions.procedure === 'zero')
        && modularOptions.distance_type === 'imprecise';

    const whiteStack = document.getElementById('white-cards');
    if (whiteStack) {
        whiteStack.style.display = noBlankCards ? 'none' : 'block';
    }

    const stacksContainer = document.querySelector('.card-stacks');
    if (stacksContainer) {
        let noBlankHint = document.getElementById('method-no-blank-hint');
        if (!noBlankHint) {
            noBlankHint = document.createElement('div');
            noBlankHint.id = 'method-no-blank-hint';
            noBlankHint.style.fontSize = '0.8rem';
            noBlankHint.style.color = '#6f7784';
            noBlankHint.style.marginTop = '0.5rem';
            stacksContainer.appendChild(noBlankHint);
        }
        if (selectedMethod === 'modular_srf' && modularOptions?.procedure === 'direct') {
            noBlankHint.textContent = 'Direct-ratio module does not use blank cards.';
        } else if (modularImpreciseDistance) {
            noBlankHint.textContent = 'Modular imprecise distance input is active: visual blank cards are disabled and ignored.';
        } else if (methodForInputs === 'imprecise_srf') {
            noBlankHint.textContent = 'Imprecise SRF uses interval gap inputs; visual blank cards are locked and ignored.';
        } else if (methodForInputs === 'belief_degree_imprecise_srf') {
            noBlankHint.textContent = 'Belief-degree imprecise SRF uses belief distributions for gaps; visual blank cards are locked and ignored.';
        } else if (methodForInputs === 'wap') {
            noBlankHint.textContent = 'WAP does not use blank cards.';
        } else if (methodForInputs === 'hfl_srf') {
            noBlankHint.textContent = 'HFL-SRF does not use blank cards.';
        } else {
            noBlankHint.textContent = '';
        }
        noBlankHint.style.display = noBlankCards ? 'block' : 'none';
    }

    if (noBlankCards) {
        document.querySelectorAll('.drop-zone .card.white').forEach(card => card.remove());
    }
}


function updateGridState() {
    /*
    This function updates the grid state by performing necessary layout adjustments and UI updates.
     */
    enforceMethodSpecificCardRules();
    reorderRows(); // Shift cards to fill vertical gaps
    showWatermarkIfEmpty(); // Show watermark if the drop zone is empty
    updateRanks(); // Update the ranks displayed underneath the cards

    // Update the input fields for z and e values depending on the selected method
    ensureModularQuestionnairePanel()
    renderZInputs()
    renderEInputs()
    updateSamplingSizeVisibility()
    ensureOptionalConstraintsPanel()
    validateMethodInputsAndToggleRun()
}


function reorderRows() {
    /*
    This function shifts cards upwards in case there are some new empty spots due to cards being
    moved somewhere else or being deleted. This is just for visual appeal and does not affect the
    calculation logic in the backend.
    */

    // identify the drop zone element and count the number of columns (empty and non-empty)
    const dropZone = document.querySelector('.drop-zone');
    const columnCount = (window.getComputedStyle(dropZone).getPropertyValue('grid-template-columns').match(/px/g) || []).length;

    for (let col = 1; col <= columnCount; col++) {
        // get all cards in a specified column
        var cardsInColumn = Array.from(dropZone.querySelectorAll(`.card[style*="grid-column-start: ${col};"]`));

        // sort non-empty cards to the top
        var row = 0;
        cardsInColumn.sort((a, b) => (b.textContent.trim() === '') - (a.textContent.trim() === ''));
        for (var card of cardsInColumn) {
            row++;
            card.style.gridRowStart = row;
            card.style.gridColumnStart = col;
        }
    }
}


function showWatermarkIfEmpty() {
    // This function shows a watermark at the center of the drop zone in case it is empty

    // identify the drop zone and the placeholder of the watermark
    const dropZone = document.querySelector('.drop-zone');
    const placeholder = document.querySelector('.placeholder-text');

    // show watermark if the drop zone is empty, otherwise hide it
    if (dropZone.children.length === 0) {
        placeholder.style.display = 'block';
    } else {
        placeholder.style.display = 'none';
    }
}


function updateRanks() {
    // this function updates the ranks that are displayed underneath the cards

    // identify the drop zone and the grid underneath that displays the ranks
    const dropZone = document.querySelector('.drop-zone');
    const ranksContainer = document.getElementById('ranks-container');

    // count the columns and reset the rank counter
    var columnCount = (window.getComputedStyle(dropZone).getPropertyValue('grid-template-columns').match(/px/g) || []).length;
    var rank = 0;
    var dict_whites = {};

    // clear previous ranks
    ranksContainer.innerHTML = '';

    // assign ranks for every non-empty column with criterion cards
    // ranks are assigned from right to left (i.e., rank 1 corresponds to the least importance)
    for (var col = columnCount; col >= 1; col--) {
        var row = 1;
        if (isCellOccupied(dropZone, row, col)) {
            var card = dropZone.querySelector(`.card[style*="grid-row-start: ${row};"][style*="grid-column-start: ${col};"]`);
            if (card.classList.contains('criterion')) {
                rank++;
                const rankElement = document.createElement('div');
                rankElement.classList.add('rank');
                rankElement.textContent = `Rank ${rank}`;

                rankElement.style.gridColumn = col;
                rankElement.style.gridRow = row;

                ranksContainer.appendChild(rankElement);

                dict_whites[rank] = 0;
            } else if (rank > 0) {
                dict_whites[rank] += 1;
            }
        }
    }

    num_ranks = rank;
    white_cards = dict_whites;
}


function renderZInputs() {
    const numRanks = num_ranks;
    const zContainer = document.getElementById("z_value_query");
    const addInputsContainer = document.getElementById("additional-inputs");
    rememberMethodInputs(zContainer);
    const selectedMethod = document.getElementById("srf_method").value;
    const modularOptions = selectedMethod === 'modular_srf' ? collectModularOptionsFromDom() : null;
    const methodForInputs = selectedMethod === 'modular_srf'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;
    let zMode = methodForInputs;
    if (selectedMethod === 'modular_srf' && modularOptions) {
        if (modularOptions.procedure === 'direct') {
            zMode = 'wap';
        } else if (modularOptions.procedure === 'zero') {
            zMode = 'srf_ii';
        } else if (modularOptions.z_type === 'imprecise') {
            if (modularOptions.probability === 'yes') {
                zMode = 'belief_degree_imprecise_srf';
            } else if (modularOptions.z_format === 'fuzzy') {
                zMode = 'hfl_srf';
            } else {
                zMode = 'imprecise_srf';
            }
        } else {
            zMode = 'srf';
        }
    }

    zContainer.style.display = 'flex';

    // Set the parent containers' style
    if (zMode === 'imprecise_srf' || zMode === 'hfl_srf') {
        Object.assign(addInputsContainer.style, {
            maxWidth: "60rem",
            width: "100%"
        });
    } else if (zMode === 'belief_degree_imprecise_srf') {
        Object.assign(addInputsContainer.style, {
            maxWidth: "61.5rem",
            width: "100%"
        });
    } else {
        Object.assign(addInputsContainer.style, {
            maxWidth: "53rem",
            width: "100%"
        });
    }

    if (zMode === 'wap') {
        Object.assign(zContainer.style, {
            flexDirection: "column",
            gap: "0.3rem",
            marginBottom: "2rem"
        });
    } else if (zMode === 'hfl_srf') {
        Object.assign(zContainer.style, {
            flexDirection: "column",
            gap: "0.3rem",
            marginBottom: "2rem"
        });
    } else {
        Object.assign(zContainer.style, {
            flexDirection: "row",
            gap: "0",
            marginBottom: "0"
        });
    }

    if (zMode === 'srf_ii') {
        zContainer.innerHTML = '';
        zContainer.style.display = 'none';
    } else if (zMode === 'wap') {
        if (isNaN(numRanks) || numRanks < 2) {
            zContainer.innerHTML = "<label>Waiting for a valid number of ranks (>= 2) ...</label>";
            zContainer.style.marginBottom = "0.1rem";
            return;
        }

        const isModularDirect = selectedMethod === 'modular_srf' && modularOptions?.procedure === 'direct';
        const directPrecise = isModularDirect && modularOptions?.distance_type === 'precise';
        const directFuzzy = isModularDirect
            && modularOptions?.distance_type === 'imprecise'
            && modularOptions?.distance_format === 'fuzzy';

        let html = '';
        if (directPrecise) {
            html = `<label>
                Provide one exact successive ratio z<sub>r</sub> for each pair of ranks.
            </label>`;
        } else if (directFuzzy) {
            html = `<label>
                Choose hesitant fuzzy term intervals for each successive ratio z<sub>r</sub> (term scale 1 to 10).
            </label>`;
        } else {
            html = `<label>
                By what factor (min-max interval) does each ex aequo group outweigh the next lower-ranked group (z<sub>r</sub> values)?
            </label>`;
        }

        // Header row
        if (directPrecise) {
            html += `
            <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem; 
                        align-items: center; margin-bottom: 0.8rem; font-weight: 500; font-size: 1.05rem;
                        padding-bottom: 0.3rem; border-bottom: 1px dashed #c8c8c8; width: 23.5rem">
                <div style="width: 16rem;"><i class="fa-solid fa-list-ol"></i> Pair of successive ranks</div>
                <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-ruler-horizontal"></i> Exact z</div>
            </div>`;
        } else if (directFuzzy) {
            html += `
            <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem; 
                        align-items: center; margin-bottom: 0.8rem; font-weight: 500; font-size: 1.05rem;
                        padding-bottom: 0.3rem; border-bottom: 1px dashed #c8c8c8; width: 30rem">
                <div style="width: 16rem;"><i class="fa-solid fa-list-ol"></i> Pair of successive ranks</div>
                <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-down-to-line"></i> smaller term</div>
                <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-up-to-line"></i> larger term</div>
            </div>`;
        } else {
            html += `
            <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem; 
                        align-items: center; margin-bottom: 0.8rem; font-weight: 500; font-size: 1.05rem;
                        padding-bottom: 0.3rem; border-bottom: 1px dashed #c8c8c8; width: 30rem">
                <div style="width: 16rem;"><i class="fa-solid fa-list-ol"></i> Pair of successive ranks</div>
                <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-down-to-line"></i> Min</div>
                <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-up-to-line"></i> Max</div>
            </div>`;
        }

        for (let i = 1; i < numRanks; i++) {
            if (directPrecise) {
                html += `
                <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem;">
                    <label style="width: 16rem; border-bottom: 1px dashed #e0e0e0;">${i}.   Rank ${i + 1} / Rank ${i}</label>
                    <input type="number" name="zexact-${i}" id="zexact-${i}" 
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="0.1" min="1.1" max="100" value="1.7"
                           placeholder="z">
                </div>
                `;
            } else if (directFuzzy) {
                html += `
                <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem;">
                    <label style="width: 16rem; border-bottom: 1px dashed #e0e0e0;">${i}.   Rank ${i + 1} / Rank ${i}</label>
                    <select name="wap-hfl-zmin-${i}" id="wap-hfl-zmin-${i}" class="labelmaxmin form-control">
                        ${buildHflTermOptions(HFL_Z_TERM_DEFS, 4)}
                    </select>
                    <select name="wap-hfl-zmax-${i}" id="wap-hfl-zmax-${i}" class="labelmaxmin form-control">
                        ${buildHflTermOptions(HFL_Z_TERM_DEFS, 7)}
                    </select>
                </div>
                `;
            } else {
                html += `
                <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem;">
                    <label style="width: 16rem; border-bottom: 1px dashed #e0e0e0;">${i}.   Rank ${i + 1} / Rank ${i}</label>
                    <input type="number" name="zmin-${i}" id="zmin-${i}" 
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="0.1" min="1.1" max="100" value="1.3"
                           placeholder="zmin">
                    <input type="number" name="zmax-${i}" id="zmax-${i}" 
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="0.1" min="1.1" max="100" value="2.1"
                           placeholder="zmax">
                </div>
                `;
            }
        }

        if (directFuzzy) {
            html += buildHflDefinitionsHtml();
        }
        zContainer.innerHTML = html;
    } else if (zMode === 'imprecise_srf') {
        zContainer.innerHTML = `<label for="z-value">
                    By what factor does the most important ex aequo group outweigh the least important one (z value)?
                </label>
                <div style="display: flex; flex-direction: row; gap: 0.3rem">
                    <div style="display: flex; flex-direction: column; align-items: center">
                        <input type="number" id="zmin" name="z-value-min"
                               class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                               step=0.5  min=1.5 max=1000 value=5.5
                               placeholder="Enter a value">
                        <span style="font-size: 0.75rem;">min</span>
                    </div>
                    
                    <div style="display: flex; flex-direction: column; align-items: center">
                        <input type="number" id="zmax" name="z-value-max"
                               class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                               step=0.5  min=1.5 max=1000 value=7.5
                               placeholder="Enter a value">   
                        <span style="font-size: 0.75rem;">max</span>
                    </div>                             
                </div>`;
    } else if (zMode === 'belief_degree_imprecise_srf') {
        zContainer.innerHTML = `
            <label for="z-value">
                Global ratio z belief distribution
                <span title="Belief degree beta is your confidence mass for each z candidate. Betas must sum to 1."
                      style="cursor: help; color: #5a6372;">&#9432;</span>
            </label>

            <div style="display: flex; flex-direction: column; gap: 0.25rem; border: 1px dashed #d3d9e2; border-radius: 0.45rem; padding: 0.55rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr auto; gap: 0.4rem; font-size: 0.82rem; color: #5a6372;">
                    <div>z (ratio)</div>
                    <div>beta</div>
                    <div></div>
                </div>
                <div id="belief-z-rows">
                    ${buildBeliefRowHtml('z', 1, 5.5, 0.4)}
                    ${buildBeliefRowHtml('z', 2, 7.5, 0.6)}
                </div>
                <div style="display:flex; gap:0.35rem; align-items:center; flex-wrap: wrap;">
                    <button type="button" class="btn btn-outline-secondary btn-sm" onclick="addBeliefRow('z')"
                            style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                        <i class="fa-solid fa-plus"></i> Add row
                    </button>
                    <button type="button" class="btn btn-outline-secondary btn-sm" onclick="normalizeBeliefRows('z')"
                            style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                        <i class="fa-solid fa-scale-balanced"></i> Normalize betas
                    </button>
                    <button type="button" class="btn btn-outline-secondary btn-sm" onclick="clearBeliefRows('z')"
                            style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                        <i class="fa-solid fa-broom"></i> Clear
                    </button>
                    <span id="belief-z-sum" style="margin-left:auto; font-size:0.82rem;">sum(beta) = 1.0000</span>
                </div>
            </div>
        `;
    } else if (zMode === 'hfl_srf') {
        const eMinDefault = 4;
        const eMaxDefault = 7;
        zContainer.innerHTML = `<label for="hfl-emin">
                    Choose the hesitant fuzzy interval for global importance contrast z (term scale 1 to 10):
                </label>
                <div style="font-size: 0.8rem; color: #6f7784; margin-bottom: 0.35rem;">
                    Global z terms use fuzzy sets on 1..10. Select lower and upper term bounds.
                </div>
                <div style="display: flex; flex-direction: row; gap: 0.3rem">
                    <div style="display: flex; flex-direction: column; align-items: center">
                        <select id="hfl-emin-term" name="hfl-emin-term" class="labelmaxmin form-control">
                            ${buildHflTermOptions(HFL_Z_TERM_DEFS, eMinDefault)}
                        </select>
                        <span style="font-size: 0.75rem;">smaller importance contrast</span>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center">
                        <select id="hfl-emax-term" name="hfl-emax-term" class="labelmaxmin form-control">
                            ${buildHflTermOptions(HFL_Z_TERM_DEFS, eMaxDefault)}
                        </select>
                        <span style="font-size: 0.75rem;">larger importance contrast</span>
                    </div>
                </div>
                ${buildHflDefinitionsHtml()}`;
    } else {
        zContainer.innerHTML = `<label for="z-value">
                    By what factor does the most important ex aequo group outweigh the least important one (z value)?
                </label>
                <input type="number" id="z-value" name="z-value"
                       class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                       step=0.5  min=1.5 max=1000 value=6.5
                       placeholder="Enter a value">`;
    }

    if (zMode === 'belief_degree_imprecise_srf') {
        ensureRememberedBeliefRows('z');
    }
    restoreMethodInputs(zContainer);
}


function renderEInputs() {
    const numRanks = num_ranks;
    const numWhites = Object.values(white_cards).reduce((acc, val) => acc + val, 0);
    const eContainer = document.getElementById("e0_value_query");
    rememberMethodInputs(eContainer);
    const selectedMethod = document.getElementById("srf_method").value;
    const modularOptions = selectedMethod === 'modular_srf' ? collectModularOptionsFromDom() : null;
    const methodForInputs = selectedMethod === 'modular_srf'
        ? deriveModularInputProfile(modularOptions)
        : selectedMethod;
    let eMode = methodForInputs;
    if (selectedMethod === 'modular_srf' && modularOptions) {
        if (modularOptions.procedure === 'direct') {
            eMode = 'wap';
        } else if (modularOptions.procedure === 'zero') {
            if (modularOptions.distance_type === 'imprecise') {
                if (modularOptions.probability === 'yes') {
                    eMode = 'belief_degree_imprecise_srf';
                } else if (modularOptions.distance_format === 'fuzzy') {
                    eMode = 'hfl_srf';
                } else {
                    eMode = 'imprecise_srf';
                }
            } else {
                eMode = 'srf_ii';
            }
        } else if (modularOptions.distance_type === 'imprecise') {
            if (modularOptions.probability === 'yes') {
                eMode = 'belief_degree_imprecise_srf';
            } else if (modularOptions.distance_format === 'fuzzy') {
                eMode = 'hfl_srf';
            } else {
                eMode = 'imprecise_srf';
            }
        } else {
            eMode = 'srf';
        }
    }

    // Set the parent containers' style
    if (eMode === 'imprecise_srf' || eMode === 'hfl_srf') {
        Object.assign(eContainer.style, {
            flexDirection: "column",
            gap: "0.3rem",
            marginBottom: "2rem"
        });
    } else if (eMode === 'belief_degree_imprecise_srf') {
        Object.assign(eContainer.style, {
            flexDirection: "column",
            gap: "0.3rem",
            marginBottom: "2rem"
        });
    } else {
        Object.assign(eContainer.style, {
            flexDirection: "row",
            gap: "0",
            marginBottom: "0"
        });
    }

    const includeZeroE0 = selectedMethod === 'modular_srf' && modularOptions?.procedure === 'zero';
    const showAllGapInputs = selectedMethod === 'modular_srf' || eMode === 'imprecise_srf';
    const zeroImprecise = includeZeroE0 && modularOptions?.distance_type === 'imprecise';
    const zeroUseProb = zeroImprecise && modularOptions?.probability === 'yes';
    const zeroFuzzy = zeroImprecise && !zeroUseProb && modularOptions?.distance_format === 'fuzzy';
    const zeroInterval = zeroImprecise && !zeroUseProb && modularOptions?.distance_format !== 'fuzzy';

    const buildZeroE0PreciseBlock = () => `
        <div style="margin-bottom:0.55rem;">
            <label for="e0-value">
                Zero-criterion anchor: number of blank cards from least-important rank to zero criterion (e<sub>0</sub>)
            </label>
            <input type="number" id="e0-value" name="e0-value"
                   class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                   step=1 min=0 max=999 value=4
                   placeholder="Enter e0">
        </div>
    `;
    const buildZeroE0IntervalBlock = () => `
        <div style="margin-bottom:0.55rem;">
            <label>
                Zero-criterion anchor interval e<sub>0</sub> (blank cards from least-important rank to zero criterion)
            </label>
            <div style="display:flex; gap:0.35rem; align-items:flex-start; max-width:24rem;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <input type="number" id="e0min-value" name="e0min-value"
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="1" min="0" max="999" value="2"
                           placeholder="e0_min">
                    <span style="font-size:0.75rem;">min</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <input type="number" id="e0max-value" name="e0max-value"
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="1" min="0" max="999" value="6"
                           placeholder="e0_max">
                    <span style="font-size:0.75rem;">max</span>
                </div>
            </div>
        </div>
    `;
    const buildZeroE0HflBlock = () => `
        <div style="margin-bottom:0.55rem;">
            <label>
                Zero-criterion anchor HFL interval e<sub>0</sub> (card-term scale 1 to 5)
            </label>
            <div style="display:flex; gap:0.35rem; align-items:flex-start; max-width:24rem;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <select id="e0-rmin-term" name="e0-rmin-term" class="labelmaxmin form-control">
                        ${buildHflTermOptions(HFL_CARD_TERM_DEFS, 2)}
                    </select>
                    <span style="font-size:0.75rem;">smaller gap</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <select id="e0-rmax-term" name="e0-rmax-term" class="labelmaxmin form-control">
                        ${buildHflTermOptions(HFL_CARD_TERM_DEFS, 4)}
                    </select>
                    <span style="font-size:0.75rem;">larger gap</span>
                </div>
            </div>
        </div>
    `;
    const buildZeroE0BeliefBlock = () => `
        <div style="margin-bottom:0.55rem;">
            <label>
                Zero-criterion anchor belief distribution for e<sub>0</sub>
                <span title="Define belief degree for e0 values. Betas must sum to 1."
                      style="cursor: help; color: #5a6372;">&#9432;</span>
            </label>
            <div style="display:flex; flex-direction:column; gap:0.25rem; border:1px dashed #d3d9e2; border-radius:0.45rem; padding:0.55rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr auto; gap: 0.4rem; font-size: 0.82rem; color: #5a6372;">
                    <div>e0 (# blank cards)</div>
                    <div>beta</div>
                    <div></div>
                </div>
                <div id="belief-gap-0-rows">
                    ${buildBeliefRowHtml('e', 1, 2, 0.5, 0, false)}
                    ${buildBeliefRowHtml('e', 2, 5, 0.5, 0, false)}
                </div>
                <div style="display:flex; gap:0.35rem; align-items:center; flex-wrap: wrap;">
                    <button type="button" class="btn btn-outline-secondary btn-sm" onclick="addBeliefRow('e', 0)"
                            style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                        <i class="fa-solid fa-plus"></i> Add row
                    </button>
                    <button type="button" class="btn btn-outline-secondary btn-sm" onclick="normalizeBeliefRows('e', 0)"
                            style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                        <i class="fa-solid fa-scale-balanced"></i> Normalize betas
                    </button>
                    <button type="button" class="btn btn-outline-secondary btn-sm" onclick="clearBeliefRows('e', 0)"
                            style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                        <i class="fa-solid fa-broom"></i> Clear
                    </button>
                    <span id="belief-gap-0-sum" style="margin-left:auto; font-size:0.82rem;">sum(beta) = 1.0000</span>
                </div>
            </div>
        </div>
    `;

    const buildZeroE0Block = () => {
        if (!includeZeroE0) return '';
        if (zeroUseProb) return buildZeroE0BeliefBlock();
        if (zeroFuzzy) return buildZeroE0HflBlock();
        if (zeroInterval) return buildZeroE0IntervalBlock();
        return buildZeroE0PreciseBlock();
    };

    if (eMode === 'srf') {
        eContainer.innerHTML = `<label>
            Distances between ranks are treated as precise from the deck-of-cards arrangement.
        </label>`;
        eContainer.style.marginBottom = "0.1rem";
    } else if (eMode === 'wap') {
        eContainer.innerHTML = `<label>
            Direct-ratio module does not use blank cards or e<sub>0</sub>.
        </label>`;
        eContainer.style.marginBottom = "0.1rem";
    } else if (eMode === 'imprecise_srf') {
        if (!showAllGapInputs && (isNaN(numWhites) || numWhites < 1)) {
            eContainer.innerHTML = "<label>No blank cards have been inserted yet ...</label>";
            eContainer.style.marginBottom = "0.1rem";
            return;
        }

        let html = `${includeZeroE0 ? buildZeroE0Block() : ''}<label>
            Specify the interval (min-max) of blank cards (e<sub>r</sub>) between successive ex-aequo groups:
        </label>`;

        // Header row
        html += `
        <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.3rem; 
                    align-items: center; margin-bottom: 0.5rem; font-weight: 500; font-size: 1.05rem;
                    padding-bottom: 0.3rem; border-bottom: 1px dashed #c8c8c8; width: 30rem">
            <div style="width: 16rem;"><i class="fa-solid fa-list-ol"></i> Pair of successive ranks</div>
            <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-down-to-line"></i> Min</div>
            <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-up-to-line"></i> Max</div>
        </div>`;

        var counter = 0;
        for (let i = 1; i < numRanks; i++) {
            if (showAllGapInputs || white_cards[i] > 0) {
                html += `
                <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem;">
                    <label style="width: 16rem; border-bottom: 1px dashed #e0e0e0;">${++counter}.   Rank ${i} & Rank ${i + 1}</label>
                    <input type="number" name="emin-${i}" id="emin-${i}" 
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="1" min="0" max="100" value="${Math.max(0, white_cards[i] ?? 0)}"
                           placeholder="e_min">
                    <input type="number" name="emax-${i}" id="emax-${i}" 
                           class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                           step="1" min="0" max="100" value="${Math.max(0, (white_cards[i] ?? 0) + 2)}"
                           placeholder="e_max">
                </div>
                `;
            }
        }

        html += `</table>`;
        eContainer.innerHTML = html;
    } else if (eMode === 'belief_degree_imprecise_srf') {
        let html = `${includeZeroE0 ? buildZeroE0Block() : ''}<label style="margin-top: 0.8rem; margin-bottom: 0.5rem">
            Belief distributions for blank cards between ranks
            <span title="For each rank gap, define a belief distribution over the number of blank cards. Betas must sum to 1."
                  style="cursor: help; color: #5a6372;">&#9432;</span>
        </label>`;

        html += `<div style="max-height: 24rem; overflow-y: auto; border: 1px dashed #d3d9e2; border-radius: 0.45rem; padding: 0.55rem;">`;

        for (let i = 1; i < numRanks; i++) {
            const observedGap = Math.max(0, parseInt(String(white_cards[i] ?? 0), 10));
            const initialRows = `${buildBeliefRowHtml('e', 1, observedGap, 0.4, i, false)}
                   ${buildBeliefRowHtml('e', 2, observedGap + 1, 0.6, i, false)}`;

            html += `
                <details open style="border:1px solid #e5e9ef; border-radius:0.4rem; padding:0.35rem 0.45rem; margin-bottom:0.45rem; background:#fff;">
                    <summary style="cursor:pointer; font-weight:500;">Gap between Rank ${i} and Rank ${i + 1}</summary>
                    <div style="margin-top:0.45rem;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr auto; gap: 0.4rem; font-size: 0.82rem; color: #5a6372; margin-bottom: 0.15rem;">
                            <div>e (# blank cards)</div>
                            <div>beta</div>
                            <div></div>
                        </div>
                        <div id="belief-gap-${i}-rows">${initialRows}</div>
                        <div style="display:flex; gap:0.35rem; align-items:center; flex-wrap: wrap; margin-top: 0.25rem;">
                            <button type="button" class="btn btn-outline-secondary btn-sm"
                                    onclick="addBeliefRow('e', ${i})"
                                    style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                                <i class="fa-solid fa-plus"></i> Add row
                            </button>
                            <button type="button" class="btn btn-outline-secondary btn-sm"
                                    onclick="normalizeBeliefRows('e', ${i})"
                                    style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                                <i class="fa-solid fa-scale-balanced"></i> Normalize betas
                            </button>
                            <button type="button" class="btn btn-outline-secondary btn-sm"
                                    onclick="useObservedGapBelief(${i})"
                                    style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                                <i class="fa-solid fa-wand-magic-sparkles"></i> Use observed deck gap
                            </button>
                            <button type="button" class="btn btn-outline-secondary btn-sm"
                                    onclick="clearBeliefRows('e', ${i})"
                                    style="border-radius:0.35rem; border:1px dashed; line-height:1.5;">
                                <i class="fa-solid fa-broom"></i> Clear
                            </button>
                            <span id="belief-gap-${i}-sum" style="margin-left:auto; font-size:0.82rem;">sum(beta) = 1.0000</span>
                        </div>
                    </div>
                </details>
            `;
        }

        html += `</div>`;
        eContainer.innerHTML = html;
    } else if (eMode === 'hfl_srf') {
        if (isNaN(numRanks) || numRanks < 2) {
            eContainer.innerHTML = "<label>Waiting for a valid number of ranks (>= 2) ...</label>";
            eContainer.style.marginBottom = "0.1rem";
            return;
        }

        let html = `${includeZeroE0 ? buildZeroE0Block() : ''}<label style="margin-top: 0.8rem; margin-bottom: 0.5rem">
            Specify the HFL interval [tau<sub>s</sub><sup>low</sup>, tau<sub>s</sub><sup>upp</sup>] for the importance gap between each pair of successive ranks (card-term scale 1 to 5):
        </label>`;

        html += `<div style="font-size: 0.8rem; color: #6f7784; margin-bottom: 0.35rem;">
            Rank-gap terms use fuzzy sets on 1..5.
        </div>`;

        html += `
        <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.3rem; 
                    align-items: center; margin-bottom: 0.5rem; font-weight: 500; font-size: 1.05rem;
                    padding-bottom: 0.3rem; border-bottom: 1px dashed #c8c8c8; width: 30rem">
            <div style="width: 16rem;"><i class="fa-solid fa-list-ol"></i> Pair of successive ranks</div>
            <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-down-to-line"></i> smaller gap</div>
            <div style="width: 6.5rem; text-align: center;"><i class="fa-solid fa-arrows-up-to-line"></i> larger gap</div>
        </div>`;

        for (let i = 1; i < numRanks; i++) {
            const baseGap = 2;
            const upperGap = 4;
            html += `
                <div style="display: flex; justify-content: flex-end; margin-left: auto; gap: 0.5rem;">
                    <label style="width: 16rem; border-bottom: 1px dashed #e0e0e0;">${i}. Preference interval for Rank ${i + 1} over Rank ${i}</label>
                    <select name="hfl-rmin-${i}" id="hfl-rmin-${i}" class="labelmaxmin form-control">
                        ${buildHflTermOptions(HFL_CARD_TERM_DEFS, baseGap)}
                    </select>
                    <select name="hfl-rmax-${i}" id="hfl-rmax-${i}" class="labelmaxmin form-control">
                        ${buildHflTermOptions(HFL_CARD_TERM_DEFS, upperGap)}
                    </select>
                </div>
            `;
        }

        eContainer.innerHTML = html;
    } else {
        eContainer.innerHTML = `<label for="e0-value">
                    How many blank cards separate the least important ex aequo from a "zero criterion" (e<sub>0</sub> value)?
                </label>
                <input type="number" id="e0-value" name="e0-value"
                       class="labelmaxmin form-control" oninput="enforceMinMaxLimits(event)"
                       step=1 min=0 max=999 value=4
                       placeholder="Enter a value">`;
    }

    if (eMode === 'belief_degree_imprecise_srf') {
        if (includeZeroE0 && zeroUseProb) {
            ensureRememberedBeliefRows('e', 0);
        }
        for (let i = 1; i < numRanks; i++) {
            ensureRememberedBeliefRows('e', i);
        }
    }
    restoreMethodInputs(eContainer);
}


function addPair(index, paramType) {
    let valuesGroup;
    let betasGroup;
    let newIndex;
    if (paramType === 'e-value') {
        valuesGroup = document.getElementById(`e-value-${index}-1`).closest('.e-values-group');
        betasGroup = document.getElementById(`e-beta-${index}-1`).closest('.e-betas-group');
        newIndex = document.querySelectorAll("input[id^='" + `e-value-${index}-` + "']").length + 1;
    } else {
        valuesGroup = document.getElementById(`z-value-1`).closest('.z-values-group');
        betasGroup = document.getElementById(`z-beta-1`).closest('.z-betas-group');
        newIndex = document.querySelectorAll("input[id^='" + `z-value-` + "']").length + 1;
    }

    const newEInput = document.createElement('input');
    newEInput.type = "number";
    newEInput.name = (paramType === 'e-value') ? `e-value-${index}-${newIndex}` : `z-value-${newIndex}`;
    newEInput.id = (paramType === 'e-value') ? `e-value-${index}-${newIndex}` : `z-value-${newIndex}`;
    newEInput.className = "labelmaxmin form-control";
    newEInput.value = (paramType === 'e-value') ? "1" : "6.5";
    newEInput.step = (paramType === 'e-value') ? "1" : "0.5";
    newEInput.min = (paramType === 'e-value') ? "0" : "1.5";
    newEInput.max = (paramType === 'e-value') ? "100": "1000";
    newEInput.placeholder = (paramType === 'e-value') ? "e_r" : "z_i";
    newEInput.oninput = enforceMinMaxLimits;

    const newBInput = document.createElement('input');
    newBInput.type = "number";
    newBInput.name = (paramType === 'e-value') ? `e-beta-${index}-${newIndex}` : `z-beta-${newIndex}`;
    newBInput.id = (paramType === 'e-value') ? `e-beta-${index}-${newIndex}` : `z-beta-${newIndex}`;
    newBInput.className = "labelmaxmin form-control";
    newBInput.value = "0.25";
    newBInput.step = "0.05";
    newBInput.min = "0";
    newBInput.max = "1";
    newBInput.placeholder = "beta";
    newBInput.oninput = enforceMinMaxLimits;

    valuesGroup.appendChild(newEInput);
    betasGroup.appendChild(newBInput);
    validateMethodInputsAndToggleRun();
}


document.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.closest('#additional-inputs')) {
        validateMethodInputsAndToggleRun();
    }
});

document.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.closest('#additional-inputs')) {
        syncOptionalConstraintPanels();
        validateMethodInputsAndToggleRun();
    }
});
