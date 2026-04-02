async function exportToXLSX(filename = 'simos_method_results.xlsx') {
    if (!Array.isArray(simos_calc_results) || simos_calc_results.length === 0) {
        alert('Please run a calculation before exporting results.');
        return;
    }

    // create a new workbook with the main criteria-weight worksheet
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(simos_calc_results), "Criteria Weights");

    try {
        const response = await fetch('/data/srf_export_payload.json', { cache: 'no-store' });
        if (response.ok) {
            const payload = await response.json();
            const legacySections = Array.isArray(payload?.records)
                ? [payload]
                : [];
            const namedSections = [
                payload?.sampling_results,
                payload?.extreme_scenarios
            ].filter(section => section && typeof section === 'object');

            [...namedSections, ...legacySections].forEach(section => {
                const records = Array.isArray(section?.records) ? section.records : [];
                if (records.length === 0) return;

                const rawSheetName = typeof section?.sheet_name === 'string' && section.sheet_name.trim()
                    ? section.sheet_name.trim()
                    : 'Variability Details';
                const sheetName = rawSheetName.slice(0, 31);
                XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(records), sheetName);
            });
        }
    } catch (error) {
        console.error('Error loading export payload:', error);
    }

    // generate an Excel file from the workbook and trigger download
    XLSX.writeFile(wb, filename);
}

const FIGURE_DOWNLOAD_CONTROL_ID = 'figure-download-controls';
const FIGURE_DOWNLOAD_SELECT_ID = 'figure-download-format';
const PNG_EXPORT_SCALE_300_PPI = 300 / 96;
const SUPPORTED_FIGURE_DOWNLOAD_FORMATS = new Set(['svg', 'png', 'jpeg', 'webp']);
const DISTRIBUTION_CHART_CONTROL_ID = 'distribution-chart-controls';
const DISTRIBUTION_CHART_SELECT_ID = 'distribution-chart-type';
let lastDistributionPlotState = null;


function ensureFigureDownloadControls() {
    let controls = document.getElementById(FIGURE_DOWNLOAD_CONTROL_ID);
    if (controls) {
        return controls;
    }

    const figureGrid = document.getElementById('results-figure-grid');
    const figureGridParent = figureGrid?.parentNode;
    if (!figureGrid || !figureGridParent) {
        return null;
    }

    controls = document.createElement('div');
    controls.id = FIGURE_DOWNLOAD_CONTROL_ID;
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', 'Figure download options');
    Object.assign(controls.style, {
        display: 'none',
        alignItems: 'center',
        gap: '0.65rem',
        flexWrap: 'wrap',
        margin: '1rem 0 0.9rem 0',
        padding: '0.7rem 0.85rem',
        border: '1px solid #dbe1e9',
        borderRadius: '0.55rem',
        background: '#f8fafc',
        width: '100%'
    });

    const label = document.createElement('label');
    label.htmlFor = FIGURE_DOWNLOAD_SELECT_ID;
    label.textContent = 'Figure downloads';
    label.style.fontWeight = '600';

    const select = document.createElement('select');
    select.id = FIGURE_DOWNLOAD_SELECT_ID;
    select.className = 'labelmaxmin form-control';
    select.setAttribute('aria-label', 'Choose figure download format');
    select.style.width = '12rem';
    select.innerHTML = `
        <option value="svg" selected>SVG</option>
        <option value="png">PNG (300 ppi)</option>
        <option value="jpeg">JPEG (300 ppi)</option>
        <option value="webp">WebP (300 ppi)</option>
    `;

    const hint = document.createElement('span');
    hint.textContent = 'This format selection applies to the download button in every plot toolbar. SVG is vector; raster exports use ~300 ppi scaling.';
    Object.assign(hint.style, {
        fontSize: '0.82rem',
        color: '#5a6372',
        flex: '1 1 20rem'
    });

    controls.append(label, select, hint);
    figureGridParent.insertBefore(controls, figureGrid);
    return controls;
}


function setFigureDownloadControlsVisible(isVisible) {
    const controls = ensureFigureDownloadControls();
    if (controls) {
        controls.style.display = isVisible ? 'flex' : 'none';
    }
}


function getSelectedFigureDownloadFormat() {
    const format = document.getElementById(FIGURE_DOWNLOAD_SELECT_ID)?.value;
    return SUPPORTED_FIGURE_DOWNLOAD_FORMATS.has(format) ? format : 'svg';
}


function getFigureDownloadOptions(filename, fallbackWidth, fallbackHeight, plotElement = null) {
    const format = getSelectedFigureDownloadFormat();
    const width = Number(plotElement?._fullLayout?.width) || fallbackWidth;
    const height = Number(plotElement?._fullLayout?.height) || fallbackHeight;

    return {
        format,
        filename,
        width,
        height,
        scale: format === 'svg' ? 1 : PNG_EXPORT_SCALE_300_PPI
    };
}


function buildPlotDownloadConfig(filename, fallbackWidth, fallbackHeight) {
    return {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage'],
        modeBarButtonsToAdd: [{
            name: 'Download figure',
            icon: Plotly.Icons.camera,
            click: (gd) => Plotly.downloadImage(
                gd,
                getFigureDownloadOptions(filename, fallbackWidth, fallbackHeight, gd)
            )
        }]
    };
}


function createTableFromDataframe(dataframe, selectedMethod = null) {
    /*
    This function creates a table based on the dataframe with calculation results, and displays it on the HTML page.
    */
    if (!Array.isArray(dataframe) || dataframe.length === 0) {
        return;
    }

    const method = selectedMethod || document.getElementById("srf_method")?.value;
    const weightDecimalPlacesRaw = document.getElementById("w-value")?.value
        ?? document.getElementById("w_value")?.value;
    const weightDecimalPlaces = Number.isFinite(Number.parseInt(weightDecimalPlacesRaw, 10))
        ? Math.max(0, Number.parseInt(weightDecimalPlacesRaw, 10))
        : 1;
    const nonCrispMethods = new Set([
        'robust_srf',
        'wap',
        'imprecise_srf',
        'belief_degree_imprecise_srf',
        'hfl_srf'
    ]);
    const hideSelectedWeightsInTable = nonCrispMethods.has(method);

    // create a table element
    const table = document.createElement('table');
    table.setAttribute('border', '1');

    const shouldFormatWeightCell = (header, row) => {
        if (!header || row?.["Rank [r]"] === "Sum" && header === 'Criteria') {
            return false;
        }
        return [
            'Weights [%]',
            'Center weight [k_center]',
            'Min weight [k_min]',
            'Max weight [k_max]',
        ].includes(header);
    };

    const formatWeightCellValue = (value) => {
        const numericValue = Number.parseFloat(value);
        return Number.isFinite(numericValue)
            ? numericValue.toFixed(weightDecimalPlaces)
            : value;
    };

    // create table header row
    let headers = Object.keys(dataframe[0]);
    if (hideSelectedWeightsInTable) {
        headers = headers.filter(header => header !== 'Weights [%]');
    }
    const headerRow = document.createElement('tr');
    headers.forEach(header => headerRow.innerHTML += `<th>${header}</th>`);
    table.appendChild(headerRow);

    // create table rows
    dataframe.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = shouldFormatWeightCell(header, row)
                ? formatWeightCellValue(row[header])
                : row[header];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    document.getElementById("results-container").append(table);
}


function buildFallbackDistributionFromSummaryRows(simos_calc_results) {
    if (!Array.isArray(simos_calc_results) || simos_calc_results.length === 0) {
        return {};
    }

    const fallback = {};
    simos_calc_results.forEach(row => {
        if (!row || row["Rank [r]"] === "Sum") return;
        const criterion = row["Criteria"];
        if (!criterion) return;

        const minVal = parseFloat(row["Min weight [k_min]"]);
        const maxVal = parseFloat(row["Max weight [k_max]"]);
        const centerVal = parseFloat(
            row["Center weight [k_center]"] ?? row["Weights [%]"] ?? row["Normalized weights [k_i]"]
        );
        const points = [minVal, centerVal, maxVal].filter(Number.isFinite);
        if (points.length >= 2) {
            fallback[criterion] = points;
        }
    });

    return fallback;
}


function hasUsableBoxDistribution(distributionByCriterion) {
    if (!distributionByCriterion || typeof distributionByCriterion !== 'object') {
        return false;
    }
    return Object.values(distributionByCriterion).some(values => (
        Array.isArray(values) && values.filter(Number.isFinite).length >= 2
    ));
}


function ensureDistributionChartControls() {
    let controls = document.getElementById(DISTRIBUTION_CHART_CONTROL_ID);
    if (controls) {
        return controls;
    }

    const header = document.querySelector('#boxplot-panel .results-figure-header');
    if (!header) {
        return null;
    }

    controls = document.createElement('div');
    controls.id = DISTRIBUTION_CHART_CONTROL_ID;
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', 'Sampling distribution chart type');
    Object.assign(controls.style, {
        display: 'none',
        alignItems: 'center',
        gap: '0.5rem',
        flexWrap: 'wrap'
    });

    const label = document.createElement('label');
    label.htmlFor = DISTRIBUTION_CHART_SELECT_ID;
    label.textContent = 'Distribution chart';
    Object.assign(label.style, {
        fontSize: '0.82rem',
        fontWeight: '600',
        color: '#425466'
    });

    const select = document.createElement('select');
    select.id = DISTRIBUTION_CHART_SELECT_ID;
    select.className = 'labelmaxmin form-control';
    select.setAttribute('aria-label', 'Choose sampling distribution chart type');
    select.style.width = '10rem';
    select.innerHTML = `
        <option value="box" selected>Box plot</option>
        <option value="violin">Violin plot</option>
    `;
    select.addEventListener('change', () => {
        if (lastDistributionPlotState?.hasDistribution) {
            renderDistributionPlot(lastDistributionPlotState);
        }
    });

    controls.append(label, select);
    const infoPopover = header.querySelector('.info-popover');
    header.insertBefore(controls, infoPopover || null);
    return controls;
}


function setDistributionChartControlsVisible(isVisible) {
    const controls = ensureDistributionChartControls();
    if (controls) {
        controls.style.display = isVisible ? 'inline-flex' : 'none';
    }
}


function getSelectedDistributionChartType() {
    const value = document.getElementById(DISTRIBUTION_CHART_SELECT_ID)?.value;
    return value === 'violin' ? 'violin' : 'box';
}


function buildDistributionTraces(distributionByCriterion, chartType) {
    return Object.keys(distributionByCriterion).sort().map(key => {
        const values = distributionByCriterion[key];
        const baseTrace = {
            x: Array(values.length).fill(key),
            y: values,
            name: key,
            showlegend: false,
            hovertemplate: 'Criterion: %{x}<br>Weight: %{y:.2f}%<extra></extra>'
        };

        if (chartType === 'violin') {
            return {
                ...baseTrace,
                type: 'violin',
                points: false,
                box: {
                    visible: false
                },
                meanline: {
                    visible: true
                },
                fillcolor: 'rgba(80, 149, 204, 0.58)',
                line: {
                    color: '#325d88',
                    width: 1.2
                },
                opacity: 0.9,
                spanmode: 'hard'
            };
        }

        return {
            ...baseTrace,
            type: 'box',
            boxpoints: false,
            boxmean: true,
            jitter: 0.5,
            whiskerwidth: 0.2,
            fillcolor: 'rgba(80, 149, 204, 0.38)',
            marker: {
                color: '#325d88',
                size: 2
            },
            line: {
                color: '#325d88',
                width: 1.1
            }
        };
    });
}


function buildSelectedWeightBarTrace(simos_calc_results) {
    const lineData = simos_calc_results.reduce((acc, item) => {
        if (item["Rank [r]"] !== "Sum") {
            const weightValue = item["Weights [%]"] ?? item["Normalized weights [k_i]"];
            acc[item["Criteria"]] = parseFloat(weightValue);
        }
        return acc;
    }, {});

    const sortedCriteria = Object.keys(lineData).sort();
    return {
        x: sortedCriteria,
        y: sortedCriteria.map(key => lineData[key]),
        type: 'bar',
        showlegend: false,
        marker: {
            color: 'rgba(50,93,136,0.72)',
            line: {
                color: 'black',
                width: 1
            }
        },
        width: 0.4,
        hovertemplate: 'Criterion: %{x}<br>Weight: %{y:.2f}%<extra></extra>'
    };
}


function renderDistributionPlot(plotState) {
    if (!plotState) return;

    const chartType = plotState.hasDistribution ? getSelectedDistributionChartType() : 'bar';
    updateBoxplotPanelCopy(plotState.hasDistribution, chartType);
    setDistributionChartControlsVisible(plotState.hasDistribution);

    const traces = plotState.hasDistribution
        ? buildDistributionTraces(plotState.transposed, chartType)
        : [buildSelectedWeightBarTrace(plotState.simos_calc_results)];

    const plotWidth = getPlotContainerWidth(plotState.container_id, 1200);
    const layout = {
        title: {
            text: plotState.hasDistribution
                ? (chartType === 'violin' ? 'Violin plot of feasible weights' : 'Box plot of feasible weights')
                : 'Weights % by Criteria',
            font: {
                weight: 'bold'
            }
        },
        yaxis: {
            title: {
                text: 'Weights %'
            },
            range: [0, null],
            showgrid: true,
            zeroline: true,
            dtick: 5,
            gridwidth: 1,
            zerolinewidth: 1.2,
            showline: true,
            mirror: 'all',
            linecolor: 'black',
            linewidth: 1,
            ticks: 'outside',
            ticklen: 6,
            tickwidth: 1,
            tickcolor: 'black'
        },
        xaxis: {
            title: {
                text: 'Criteria',
            },
            showgrid: false,
            tickmode: 'array',
            showticklabels: true,
            showline: true,
            mirror: 'all',
            linecolor: 'black',
            linewidth: 1,
            ticks: 'outside',
            ticklen: 6,
            tickwidth: 1,
            tickcolor: 'black'
        },
        showlegend: false,
        autosize: true,
        height: plotWidth < 720 ? 430 : 500,
        violingap: chartType === 'violin' ? 0.22 : undefined
    };

    const exportFilename = !plotState.hasDistribution
        ? 'SRF_bar_chart'
        : (chartType === 'violin' ? 'SRF_violin_plot' : 'SRF_box_plot');
    const config = buildPlotDownloadConfig(exportFilename, plotWidth, layout.height);

    Plotly.newPlot(plotState.container_id, traces, layout, config);
    setFigureDownloadControlsVisible(true);
}


async function plot_boxplot(simos_calc_results, noDistribution, container_id = 'boxplot') {
    setFigureCardVisibility('boxplot-panel', true);
    let transposed;
    let hasDistribution = !noDistribution;

    if (noDistribution) {
        transposed = {};
    } else {
        try {
            const response = await fetch('/data/srf_samples.json', { cache: 'no-store' });
            if (!response.ok) {
                const fallback = buildFallbackDistributionFromSummaryRows(simos_calc_results);
                if (Object.keys(fallback).length > 0) {
                    hasDistribution = true;
                    transposed = fallback;
                } else {
                    hasDistribution = false;
                    transposed = {};
                }
            }
            else {
                const data = await response.json();
                if (!Array.isArray(data) || data.length === 0) {
                    const fallback = buildFallbackDistributionFromSummaryRows(simos_calc_results);
                    if (Object.keys(fallback).length > 0) {
                        hasDistribution = true;
                        transposed = fallback;
                    } else {
                        hasDistribution = false;
                        transposed = {};
                    }
                } else {
                    const criteriaKeys = Object.keys(data[0]);
                    transposed = {};
                    for (const key of criteriaKeys) {
                        transposed[key] = data.map(sample => sample[key]);
                    }
                    if (!hasUsableBoxDistribution(transposed)) {
                        const fallback = buildFallbackDistributionFromSummaryRows(simos_calc_results);
                        if (Object.keys(fallback).length > 0) {
                            transposed = fallback;
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Error loading JSON:", error);
            const fallback = buildFallbackDistributionFromSummaryRows(simos_calc_results);
            if (Object.keys(fallback).length > 0) {
                hasDistribution = true;
                transposed = fallback;
            } else {
                hasDistribution = false;
                transposed = {};
            }
        }
    }

    lastDistributionPlotState = {
        simos_calc_results,
        transposed,
        hasDistribution,
        container_id
    };
    renderDistributionPlot(lastDistributionPlotState);
}


function setFigureCardVisibility(cardId, isVisible) {
    const card = document.getElementById(cardId);
    if (card) {
        card.style.display = isVisible ? '' : 'none';
    }
}


function getPlotContainerWidth(container_id, fallbackWidth) {
    const container = document.getElementById(container_id);
    const measuredWidth = Number(container?.clientWidth) || fallbackWidth;
    return Math.max(320, measuredWidth);
}


function resizeResultPlots() {
    ['boxplot', 'extreme_plot', 'pca_plot'].forEach(plotId => {
        const plotEl = document.getElementById(plotId);
        if (plotEl && Array.isArray(plotEl.data) && plotEl.data.length > 0) {
            Plotly.Plots.resize(plotEl);
        }
    });
}


window.addEventListener('resize', resizeResultPlots);


function updateBoxplotPanelCopy(hasDistribution, chartType = 'box') {
    const title = document.getElementById('boxplot-panel-title');
    const description = document.getElementById('boxplot-panel-description');
    const help = document.getElementById('boxplot-panel-help');

    if (!title || !description || !help) return;

    if (hasDistribution) {
        title.textContent = 'Sampling Distribution';
        if (chartType === 'violin') {
            description.textContent = 'Violin plots summarize the feasible sample cloud for each criterion. For continuous SRF models, the cloud is generated with a hit-and-run sampler targeting the uniform distribution over the admissible region, and the line inside each violin marks the sample mean.';
            help.textContent = 'The tool draws feasible solutions from the admissible SRF region. For continuous models, it uses hit-and-run sampling targeting a uniform distribution over that region; if a run requires discrete logical constraints, it falls back to exact feasible-solution exploration on the same optimization model. Each violin represents one criterion and the internal line marks the sample mean.';
        } else {
            description.textContent = 'Box plots summarize the feasible sample cloud for each criterion. For continuous SRF models, the cloud is generated with a hit-and-run sampler targeting the uniform distribution over the admissible region.';
            help.textContent = 'The tool draws feasible solutions from the admissible SRF region. For continuous models, it uses hit-and-run sampling targeting a uniform distribution over that region; if a run requires discrete logical constraints, it falls back to exact feasible-solution exploration on the same optimization model. Each box aggregates the sampled weights for one criterion without overlaying a separate selected-weight line.';
        }
        return;
    }

    title.textContent = 'Selected Weights';
    description.textContent = 'When no variability analysis is available, the figure falls back to a simple bar chart of the displayed SRF weight vector.';
    help.textContent = 'This chart is generated directly from the single SRF weight vector returned by the solver. No feasible sample cloud is used for this run, so the result is shown as bars instead of a box or violin distribution.';
}


async function plot_extreme_scenarios(noDistribution, container_id = 'extreme_plot') {
    if (noDistribution) {
        setFigureCardVisibility('extreme-panel', false);
        Plotly.purge(container_id);
        return;
    }

    try {
        const response = await fetch('/data/srf_extreme_scenarios.json', { cache: 'no-store' });
        const rows = response.ok ? await response.json() : [];
        if (!Array.isArray(rows) || rows.length === 0) {
            setFigureCardVisibility('extreme-panel', false);
            Plotly.purge(container_id);
            return;
        }

        const scenarioKey = Object.prototype.hasOwnProperty.call(rows[0], 'Scenario')
            ? 'Scenario'
            : null;
        const criteria = Object.keys(rows[0]).filter(key => key !== scenarioKey);
        if (!scenarioKey || criteria.length === 0) {
            setFigureCardVisibility('extreme-panel', false);
            Plotly.purge(container_id);
            return;
        }

        setFigureCardVisibility('extreme-panel', true);
        const scenarios = rows.map(row => row[scenarioKey]);
        const zValues = rows.map(row => criteria.map(key => {
            const value = Number.parseFloat(row[key]);
            return Number.isFinite(value) ? value : null;
        }));
        const shouldAnnotateCells = scenarios.length <= 12 && criteria.length <= 10;
        const textValues = shouldAnnotateCells
            ? zValues.map(row => row.map(value => Number.isFinite(value) ? value.toFixed(1) : ''))
            : undefined;

        const trace = {
            type: 'heatmap',
            x: criteria,
            y: scenarios,
            z: zValues,
            colorscale: 'Viridis',
            text: textValues,
            texttemplate: shouldAnnotateCells ? '%{text}' : undefined,
            textfont: shouldAnnotateCells ? { color: '#ffffff', size: 11 } : undefined,
            colorbar: {
                title: {
                    text: 'Weight %'
                }
            },
            hovertemplate: 'Scenario: %{y}<br>Criterion: %{x}<br>Weight: %{z:.2f}%<extra></extra>'
        };

        const plotWidth = getPlotContainerWidth(container_id, 960);
        const layout = {
            title: {
                text: 'Extreme scenario heatmap',
                font: {
                    weight: 'bold'
                }
            },
            xaxis: {
                title: {
                    text: 'Criteria'
                },
                showgrid: false,
                showline: true,
                mirror: 'all',
                linecolor: 'black',
                linewidth: 1,
                ticks: 'outside',
                ticklen: 6,
                tickwidth: 1,
                tickcolor: 'black'
            },
            yaxis: {
                title: {
                    text: 'Extreme scenarios',
                    standoff: plotWidth < 700 ? 18 : 28
                },
                autorange: 'reversed',
                automargin: true,
                showgrid: false,
                showline: true,
                mirror: 'all',
                linecolor: 'black',
                linewidth: 1,
                ticks: 'outside',
                ticklen: 6,
                tickwidth: 1,
                tickcolor: 'black'
            },
            autosize: true,
            height: Math.max(380, 150 + scenarios.length * 28),
            margin: {
                l: plotWidth < 700 ? 185 : 275,
                r: 70,
                t: 70,
                b: plotWidth < 700 ? 90 : 110
            }
        };

        const config = buildPlotDownloadConfig('SRF_extreme_scenarios', plotWidth, layout.height);
        Plotly.newPlot(container_id, [trace], layout, config);
        setFigureDownloadControlsVisible(true);
    } catch (error) {
        console.error("Error loading extreme-scenario JSON:", error);
        setFigureCardVisibility('extreme-panel', false);
        Plotly.purge(container_id);
    }
}


function plot_pca(noDistribution, container_id = 'pca_plot') {
    if (noDistribution) {
        setFigureCardVisibility('pca-panel', false);
        Plotly.purge(container_id);
        return;
    }

    fetch('/data/pca_output.json', { cache: 'no-store' })
        .then(response => response.ok ? response.json() : {})
        .then(data => {
            const keys = Object.keys(data);
            if (keys.length === 0) {
                setFigureCardVisibility('pca-panel', false);
                Plotly.purge(container_id);
                return;
            }

            const pc1 = keys.map(key => data[key].PC1);
            const pc2 = keys.map(key => data[key].PC2);

            // separate the selected criteria weights, vertices, and the sampled weights
            const selectedIndex = keys.indexOf("selected");
            const vertexIndices = keys
                .map((key, index) => key.startsWith("vertex_") ? index : -1)
                .filter(index => index !== -1);
            const sampleIndices = keys
                .map((key, index) => (
                    index !== selectedIndex && !key.startsWith("vertex_") ? index : -1
                ))
                .filter(index => index !== -1);

            const traceSelected = {
                x: selectedIndex === -1 ? [] : [pc1[selectedIndex]],
                y: selectedIndex === -1 ? [] : [pc2[selectedIndex]],
                mode: 'markers',
                type: 'scatter',
                name: 'Selected Criteria Weights',
                marker: {
                    color: '#D35400',
                    size: 12,
                    symbol: 'x',
                    line: {
                        color: 'black',
                        width: 1.5
                    }
                },
                text: [`Selected Criteria Weights`]
            };

            const traceVertices = {
                x: vertexIndices.map(i => pc1[i]),
                y: vertexIndices.map(i => pc2[i]),
                mode: 'markers',
                type: 'scatter',
                name: 'Vertices of the Polyhedron',
                marker: {
                    color: '#D97706',
                    size: 11,
                    symbol: 'diamond-open',
                    line: {
                        color: 'black',
                        width: 1.5
                    }
                },
                text: vertexIndices.map((_, i) => `Vertex ${i + 1}`)
            };

            const traceOthers = {
                x: sampleIndices.map(i => pc1[i]),
                y: sampleIndices.map(i => pc2[i]),
                mode: 'markers',
                type: 'scatter',
                name: 'Random Feasible Solution Samples',
                marker: {
                    color: '#325d88',
                    size: 6,
                    symbol: 'circle',
                    opacity: 0.65,
                    line: {
                        color: 'black',
                        width: 0.8
                    }
                },
                text: sampleIndices.map((_, i) => `Sample ${i + 1}`)
            };

            const trace = [];
            if (vertexIndices.length > 0) {
                trace.push(traceVertices);
            }
            if (sampleIndices.length > 0) {
                trace.push(traceOthers);
            }
            if (selectedIndex !== -1) {
                trace.push(traceSelected);
            }

            if (trace.length === 0) {
                setFigureCardVisibility('pca-panel', false);
                Plotly.purge(container_id);
                return;
            }

            setFigureCardVisibility('pca-panel', true);
            const plotWidth = getPlotContainerWidth(container_id, 700);
            const layout = {
                title: {
                    text: 'PCA of sample weights',
                    font: {
                        weight: 'bold'
                    }
                },
                xaxis: {
                    title: {
                        text: 'PC1'
                    },
                    showgrid: false,
                    zeroline: false,
                    showline: true,
                    mirror: 'all',
                    linecolor: 'black',
                    linewidth: 1,
                    ticks: 'outside',
                    ticklen: 6,
                    tickwidth: 1,
                    tickcolor: 'black'
                },
                yaxis: {
                    title: {
                        text: 'PC2',
                    },
                    showgrid: false,
                    zeroline: false,
                    showline: true,
                    mirror: 'all',
                    linecolor: 'black',
                    linewidth: 1,
                    ticks: 'outside',
                    ticklen: 6,
                    tickwidth: 1,
                    tickcolor: 'black'
                },
                showlegend: true,
                legend: {
                    x: 0.03,
                    y: 0.97,
                    xanchor: 'left',
                    yanchor: 'top',
                    bgcolor: 'rgba(255,255,255,0.5)',
                },
                autosize: true,
                height: Math.max(380, Math.min(700, Math.round(plotWidth * 0.8)))
            };

            // export plot as an image with a specified scale (higher DPI)
            const config = buildPlotDownloadConfig('SRF_pca_plot', plotWidth, layout.height);

            Plotly.newPlot(container_id, trace, layout, config);
            setFigureDownloadControlsVisible(true);
        })
        .catch(error => {
            console.error("Error loading JSON:", error);
            setFigureCardVisibility('pca-panel', false);
        });
}


ensureFigureDownloadControls();
ensureDistributionChartControls();

document.querySelector('.calculate-button')?.addEventListener('click', () => {
    lastDistributionPlotState = null;
    setFigureDownloadControlsVisible(false);
    setDistributionChartControlsVisible(false);
});
