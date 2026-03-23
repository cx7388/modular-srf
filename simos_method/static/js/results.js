function exportToXLSX(filename = 'simos_method_results.xlsx') {
    // create a new workbook with two worksheets containing normalized and non-normalized criteria weights
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(simos_calc_results), "Criteria Weights");

    // generate an Excel file from the workbook and trigger download
    XLSX.writeFile(wb, filename);
}


function createTableFromDataframe(dataframe, selectedMethod = null) {
    /*
    This function creates a table based on the dataframe with calculation results, and displays it on the HTML page.
    */
    if (!Array.isArray(dataframe) || dataframe.length === 0) {
        return;
    }

    const method = selectedMethod || document.getElementById("srf_method")?.value;
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
            td.textContent = row[header];
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


async function plot_boxplot(simos_calc_results, noDistribution, container_id = 'boxplot') {
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
                return;
            }
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

    // create Plotly traces
    const traces = Object.keys(transposed).sort().map(key => ({
        y: transposed[key],
        type: 'box',
        name: key,
        boxpoints: false,
        boxmean: true,
        jitter: 0.5,
        whiskerwidth: 0.2,
        fillcolor: 'cls',
        marker: {
            color: '#325d88',
            size: 2
        },
        line: {
            width: 1
        },
        showlegend: false,
    }));
    if (hasDistribution) {
        // add a dummy trace to represent all box plots
        traces.push({
            y: [null],
            type: 'box',
            name: 'Distribution of Criteria Weights',
            fillcolor: 'cls',
            marker: {
                color: '#325d88',
                size: 2
            },
            line: {
                width: 1
            },
            hoverinfo: 'skip',
            showlegend: true,
        });
    }

    const layout = {
        title: {
            text: 'Weights % by Criteria',
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
        showlegend: true,
        legend: {
            x: 0.03,
            y: 0.97,
            xanchor: 'left',
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.5)',
        },
        autosize: true,
        height: 500,
        width: 1200
    };

    // LINE CHART
    const lineData = simos_calc_results.reduce((acc, item) => {
        if (item["Rank [r]"] !== "Sum") {
            const weightValue = item["Weights [%]"] ?? item["Normalized weights [k_i]"];
            acc[item["Criteria"]] = parseFloat(weightValue);
        }
        return acc;
    }, {});

    // Prepare line trace
    const lineTrace = {
        x: Object.keys(lineData).sort(),      // x-axis categories
        y: Object.keys(lineData).sort().map(k => lineData[k]), // corresponding y values
        type: hasDistribution ? 'scatter' : 'bar',
        mode: hasDistribution ? 'lines+markers' : undefined,
        name: 'Selected Criteria Weights',
        line: hasDistribution ? { color: '#D35400', width: 2 } : undefined,
        marker: hasDistribution
            ? { size: 6, color: '#3e3f3a' }
            : { color: 'rgba(50,93,136,0.7)', line: { color: 'black', width: 1 } },
        width: hasDistribution ? undefined : 0.4
    };

    // Add line trace to the boxplot traces
    traces.push(lineTrace);

    // export plot as an image with a specified scale (higher DPI)
    const config = {
        toImageButtonOptions: {
            format: 'png', // set image format (png, jpeg, svg, pdf)
            height: 500,
            width: 1200,
            scale: 3,
            filename: 'SRF_box_plot'
        }
    };

    Plotly.newPlot(container_id, traces, layout, config);
}


function plot_pca(noDistribution, container_id = 'pca_plot') {
    if (noDistribution) {
        Plotly.purge('pca_plot');
        return;
    }

    fetch('/data/pca_output.json', { cache: 'no-store' })
        .then(response => response.ok ? response.json() : {})
        .then(data => {
            const keys = Object.keys(data);
            if (keys.length === 0) {
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
                Plotly.purge(container_id);
                return;
            }

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
                height: 700,
                width: 700
            };

            // export plot as an image with a specified scale (higher DPI)
            const config = {
                toImageButtonOptions: {
                    format: 'png', // set image format (png, jpeg, svg, pdf)
                    height: 700,
                    width: 700,
                    scale: 3,
                    filename: 'SRF_pca_plot'
                }
            };

            Plotly.newPlot(container_id, trace, layout, config);
        })
        .catch(error => console.error("Error loading JSON:", error));
}
