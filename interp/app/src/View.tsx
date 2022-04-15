import { focusEquals, focusIsEmpty, getAxisDimTypes, getClearedFocus, getHighlight, matchArrToHoles, matchDimOrder, range, vntPick } from "./common";
import { VeryNamedTensor, PickArgs, ViewSpecDim, LazyVeryNamedTensor, ViewSpec, NewViewManagerProps, AncestralFocus } from "./proto";
import { ViewRegistry } from "./ViewRegistry";
import { sup, inf } from "ndarray-ops";


export const DISPLAY_TYPES = ["axis", "facet"];
export const REDUCTION_TYPES = ["mean", "sum", "norm", "max", "min"]; // CM: norm is vector norm

const axisViewToColor = (a: ViewSpecDim, force = false) => {
    if (force) {
        return "var(--discouraged)";
    }
    if (REDUCTION_TYPES.indexOf(a as any) !== -1) {
        return "var(--encouraged)";
    }
    else if (DISPLAY_TYPES.indexOf(a as any) !== -1) {
        return "var(--bright_col)";
    }
    return "var(--encouraged)";
};

function capitalize(arg0: string) {
    return arg0[0].toUpperCase() + arg0.slice(1);
}


export function View(props: NewViewManagerProps) {

    const {
        lvnt,
        view,
        spec,
        vizName,
        ancestralFocus,
        focus,
        hover,
        options,
        onChangeSpec,
        unembedder,
    } = props;

    const vre = ViewRegistry.getEntry(vizName);
    const axisDimTypes = getAxisDimTypes(lvnt.dim_types, spec);
    const { viableVizes, typeMatchingVizes, narrowerVizes } = ViewRegistry.getVizes(axisDimTypes);


    // Handle user modifying the spec through the dropdown
    const handleSelectDim = (e: any, i: number) => {
        const newSpec = [...spec];
        newSpec[i] = (/^\d+$/.test(e.target.value) ?
            parseInt(e.target.value as any) :
            e.target.value);
        onChangeSpec({ viewSpec: newSpec });
    };

    const setHoverInner = (newHoverOnlyAxisDims: PickArgs, ...args: any[]) => {
        const newHover = matchArrToHoles(spec.map(x => x === "axis" || x === "facet"), focus.map(x => null), newHoverOnlyAxisDims);
        const hoverSameDimsAsFocus = newHover.every((x, i) => (x === null) === (focus[i] === null));
        // Invariant: hover is only triggered if focused already and hover is on the same dimensions
        // This prevents accidental hover from altering the children's axes in hard to undo ways
        if (newHover.every((n, i) => hover[i] === n)) {
            ; // console.log("New and old hover equal, skipping change");
        } else if (!focusIsEmpty(focus) && (hoverSameDimsAsFocus || newHover.every(x => x === null))) {
            onChangeSpec({ hover: newHover });
            // if (args.length === 0) {
            //     setHoverDeets(null);
            // } else {
            //     console.log("hover deets", args);
            //     setHoverDeets(args as any);
            // }
        } else {
            // console.log("Ignoring hover of different axes than focus", { newHoverOnlyAxisDims, newHover, hover, focus });
        }
    };

    const setFocusInner = (newFocusOnlyAxisDims: PickArgs, ...args: any[]) => {
        const newFocus =
            matchArrToHoles(spec.map(x => x === "axis" || x === "facet"), focus.map(x => null), newFocusOnlyAxisDims);
        console.log("SETTING FOCUS", { newFocus, focus, newFocusOnlyAxisDims });
        onChangeSpec({ focus: newFocus });

        // if (args.length === 0) {
        //     setHoverDeets(null);
        // } else {
        //     console.log("hover deets", args);
        //     setHoverDeets(args as any);
        // }
    };

    let vizOrFacets = (<></>);

    if (viableVizes.length === 0) {
        vizOrFacets = (<div>No views support {axisDimTypes.length} axes. Here are views supporting fewer:
            <div style={{ display: "flex", flexDirection: "row", lineHeight: "16px", fontSize: "14px", margin: "2px 0 6px 0" }}>
                {narrowerVizes.map((viz, i) => (<button key={i} style={{ padding: "0 4px", margin: "0 5px" }} onClick={() => {
                    let numToSet = axisDimTypes.length - (viz.free_dims + viz.dimTypes.length);
                    // We will set some axis to mean to reduce our dimensionality
                    // But we prefer to NOT set matching dimTypes if possible.
                    let dimTypesRemaining = viz.dimTypes.slice();
                    let disPreferredIndexes: number[] = [];
                    spec.forEach((s, i) => {
                        if (numToSet > 0 && s === "axis") {
                            let idx = dimTypesRemaining.indexOf(lvnt.dim_types[i]);
                            if (idx !== -1) {
                                disPreferredIndexes.push(i);
                                dimTypesRemaining.splice(idx, 1);
                            }
                        }
                    });
                    let newSpec = spec.map((s, i) => {
                        if (numToSet > 0 && s === "axis" && !disPreferredIndexes.includes(i)) {
                            --numToSet;
                            return "mean";
                        }
                        return s;
                    });
                    for (let i = 0; (i < dimTypesRemaining.length) && (numToSet > 0); ++i) {
                        newSpec[disPreferredIndexes[i]] = "mean";
                        --numToSet;
                    }
                    if (numToSet > 0) {
                        console.error('Failed to set leading indexes - something is wrong.');
                    }
                    onChangeSpec({ vizName: viz.name, viewSpec: newSpec });
                }} >
                    <b>{viz.name}</b><br />
                    Types: {viz.dimTypes.length > 0 ? viz.dimTypes.join(", ") : 'Any'}<br />
                    # Dims: {viz.free_dims + viz.dimTypes.length}
                </button>))}
            </div>
        </div>);
    }
    else if (!view) {
        vizOrFacets = (<div>Waiting for data</div>);
    } else {
        const dimSpecsOfView = spec.filter(x => x === "axis" || x === "facet");
        const facetDimsOfView = range(dimSpecsOfView.length).filter(i => dimSpecsOfView[i] === "facet");
        // console.log("facetdimsofview", facetDimsOfView, dimSpecsOfView, spec, view);

        // Combing focus and hover into highlight, then only send in the relevant parts
        const highlight = focus.map((x, i) => hover[i] !== null ? hover[i] : x).filter((x, i) => spec[i] === "axis");

        const Component = vre.widget;
        if (facetDimsOfView.length > 0) {
            // in here we operate on the just-view-dims (squeezed) view vnt
            const extent = [inf(view.tensor), sup(view.tensor)] as [number, number];
            const facetOptions = { ...options, extent };
            const facetFn = (dims: number[], picks: PickArgs) => {
                if (dims.length === 0) {
                    let vntHere = null;
                    vntHere = vntPick(view as VeryNamedTensor, picks);
                    if (typeMatchingVizes.indexOf(vre) !== -1) {
                        vntHere = matchDimOrder(vntHere, vre.dimTypes);
                    }

                    const setFocusInnerFacet = (dims: PickArgs, ...args: any[]) => {
                        if (dims.every(x => x === null)) {
                            return setFocusInner(picks.map(() => null), ...args);
                        }
                        return setFocusInner(matchArrToHoles(picks.map(x => x === null), picks, dims), ...args);
                    };
                    const setHoverInnerFacet = (dims: PickArgs, ...args: any[]) => {
                        if (dims.every(x => x === null)) {
                            return setHoverInner(picks.map(() => null), ...args);
                        }
                        return setHoverInner(matchArrToHoles(picks.map(x => x === null), picks, dims), ...args);
                    };
                    return (
                        <div style={{ minWidth: "100px" }} onMouseLeave={(e) => {
                            onChangeSpec({ hover: hover.map(x => null), timestamp: e.timeStamp });
                        }}>
                            <Component vnt={vntHere as any} highlight={highlight} options={facetOptions} setHover={setHoverInnerFacet} setFocus={setFocusInnerFacet} unembedder={unembedder} />
                        </div>
                    );
                }
                const direction = (["column", "row"] as ["column", "row"])[dims.length % 2];
                const otherDirection = (["column", "row"] as ["column", "row"])[(dims.length + 1) % 2];

                // Deal with dims[0] here and then recurse on rest
                return (
                    <div key={`${picks}-${dims}`} style={{ display: "flex", flexDirection: direction, flexWrap: "wrap" }}>
                        {(view as VeryNamedTensor).dim_idx_names[dims[0]].map((x, i) => {
                            const innerPicks = [...picks];
                            innerPicks[dims[0]] = i;
                            let onSelectFacet = () => {
                                setFocusInner(innerPicks);
                            };
                            return (
                                <div key={`${picks}-${innerPicks}`} style={{ margin: "5px", display: "flex", flexDirection: otherDirection }}>
                                    <div style={{ textAlign: "center", writingMode: direction === "row" ? undefined : "vertical-rl" }} onClick={onSelectFacet} >
                                        {(view as VeryNamedTensor).dim_names[dims[0]]}: {x}
                                    </div>
                                    {facetFn(dims.slice(1, dims.length), innerPicks)}
                                </div>
                            );
                        })}
                    </div>);
            };
            vizOrFacets = facetFn(facetDimsOfView, view.dim_types.map(() => null));
        } else {
            const permutedView = typeMatchingVizes.indexOf(vre) !== -1 ? matchDimOrder(view, vre.dimTypes) : view;
            vizOrFacets = (
                <Component vnt={permutedView as any}
                    highlight={highlight}
                    setHover={setHoverInner}
                    setFocus={setFocusInner}
                    options={options}
                    unembedder={unembedder} />);
        }
    }

    if (!viableVizes.map(x => x.name).includes(vizName) && viableVizes.length > 0) {
        onChangeSpec({ vizName: viableVizes[0].name });
        return (<p>Changing viz</p>);
    }

    return (
        <div style={{ padding: "0", display: "flex", flexDirection: "column" }}>
            <div className="select_bar">
                View: &nbsp;
                {lvnt.dim_idx_names.map((idx_names, i) => (
                    <AxisSelect key={i} lvnt={lvnt} i={i} idx_names={idx_names} ancestralFocus={ancestralFocus} showN={false}
                        spec={spec} handleSelectDim={handleSelectDim} />))}

            </div>
            <div className="select_bar" title="These are the chart types that work with this number of axes. The brighter ones are made for these dimension types.">
                <span>Chart type: </span>
                {viableVizes.map((viz, i) => (<button key={viz.name} className="chart_selector" style={{ outline: vizName === viz.name ? "4px solid var(--border_purple)" : "0", backgroundColor: typeMatchingVizes.indexOf(viz) === -1 ? "var(--discouraged)" : "white", margin: "7px" }} onClick={() => onChangeSpec({ vizName: viz.name })} >{viz.name}</button>))}
                {/* {viableVizes.map((viz, i) => (<button key={viz.name} style={{ padding: "0 4px", borderWidth: "1px 1px 1px 1px", outline: backgroundColor: typeMatchingVizes.indexOf(viz) === -1 ? (vizName === viz.name ? "#88c" : "#ccc") : (vizName === viz.name ? "#aaf" : "white") }} onClick={() => onChangeSpec({ vizName: viz.name })} >{viz.name}</button>))} */}
                {/* {!focus.isEmpty() && <button style={{ padding: "0 4px" }} onClick={() => onChangeSpec({ focus: focus.cleared() })}>Clear Focus</button>} */}
            </div>
            <div className="chart" onMouseLeave={(e) => {
                onChangeSpec({ hover: hover.map(x => null), timestamp: e.timeStamp });
            }}>
                {vizOrFacets}
            </div>

            <br />
            <div className="select_bar">
                <button disabled={!vizOrFacets.props["highlight"] || vizOrFacets.props["highlight"][0] === null} onClick={() => onChangeSpec({ focus: focus.map(x => null) })}>Clear Focus</button>
            </div>

        </div >);
};

interface AxisSelectProps {
    lvnt: LazyVeryNamedTensor;
    i: number;
    idx_names: string[];
    spec: ViewSpec;
    ancestralFocus: AncestralFocus;
    handleSelectDim: (e: any, i: number) => void;
    showN: boolean;
}

function AxisSelect({ lvnt, i, idx_names, spec, ancestralFocus, handleSelectDim, showN }: AxisSelectProps) {
    let name = capitalize(lvnt.dim_names[i]);
    //{viableVizes.map((viz, i) => (<button key={viz.name} className="chart_selector" style={{ outline: vizName === viz.name ? "3px solid var(--border_col)" : "0", backgroundColor: typeMatchingVizes.indexOf(viz) === -1 ? "var(--discouraged)" : "white" }} onClick={() => onChangeSpec({ vizName: viz.name })} >{viz.name}</button>))}
    let label = [(<span key={0}>{name} </span>)];
    if (lvnt.dim_names[i] !== lvnt.dim_types[i]) {
        label.push((<span key={1}>({lvnt.dim_types[i]}) </span>));
    }
    if (showN) {
        label.push((<span key={2}>n={lvnt.dim_idx_names[i].length}</span>));
    }

    let onChange = (e: any) => { handleSelectDim(e, i); };
    const followColor = axisViewToColor(spec[i], ancestralFocus.filter((x) => x.key === spec[i]).length > 0);

    // If ancestor has a specific selection, show the selection's label here - otherwise it's hard to see what's selected on the child
    let compatible = ancestralFocus.filter(x => (x.dim_idx_names.length === idx_names.length && x.dim_type === lvnt.dim_types[i]));
    let focusOptions = compatible.map((ancestorFocus) => {
        let label = (typeof ancestorFocus.pick === "number" ?
            `${lvnt.dim_idx_names[i][ancestorFocus.pick]} from ${ancestorFocus.key}` : ancestorFocus.key);
        return {
            key: ancestorFocus.key,
            value: ancestorFocus.key,
            label
        };
    });

    return (
        <div className="axis_selector" style={{ backgroundColor: followColor }} title={`These are the indexes of the tensor we're visualizing. Choose whether you want to see the mean over that axis, a specific index, plot this axis, or show one plot (facet) for each value of this axis.`}>
            {label} &nbsp;
            {/* Fixed width select ensures axes are lined up under their ancestors */}
            <select value={spec[i]?.toString()} style={{ backgroundColor: followColor, width: "100px" }} onChange={onChange} >
                <optgroup label="Display Types" style={{ backgroundColor: axisViewToColor("axis") }}>
                    {DISPLAY_TYPES.map((n) => (<option key={n} value={n}>{n}</option>))}
                </optgroup>
                <optgroup label="Reductions" style={{ backgroundColor: axisViewToColor("mean") }}>
                    {REDUCTION_TYPES.map((n) => (<option key={n} value={n} >{n}</option>))}
                </optgroup>
                <optgroup label="Link to View" style={{ backgroundColor: axisViewToColor("axis") }}>
                    {focusOptions.map((o) => (<option key={o.key} value={o.value} >{o.label}</option>))}
                </optgroup>
                <optgroup label="Indexes" style={{ backgroundColor: axisViewToColor(0) }}>
                    {idx_names.map((option, n) => (<option key={n} value={n}>{option}</option>))}
                </optgroup>
            </select></div>);

}

