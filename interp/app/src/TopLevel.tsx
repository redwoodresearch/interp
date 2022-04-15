import React from "react";
import { areAxesDifferent, getAxisDimTypes, getClearedFocus, lvntGet, range, useQuery } from "./common";
import { OldViewManagerProps as ComposableUIProps, ViewSpec, ChangeRequest, AncestorFocus, AncestralFocus, LazyVeryNamedTensor, ViewSpecDim, ComposableUIUrlState, PanelBigState, PanelUrlState, PickArgs } from "./proto";
import { View, DISPLAY_TYPES, REDUCTION_TYPES } from "./View";
import { ViewRegistry } from "./ViewRegistry";
import { useLocation, useSearchParams } from "react-router-dom";

export interface TopLevelProps {
    initialViewProps: ComposableUIProps;
    urlState: ComposableUIUrlState;
    setUrlState: (x: ComposableUIUrlState) => void;
}

interface TopLevelState {
    lvnts: LazyVeryNamedTensor[];
    panels: PanelBigState[];
    nonce: number;
    options:any
}

const isNormalViewSpecDim = (s: ViewSpecDim) => {
    return (typeof s === "number" || !s || DISPLAY_TYPES.includes(s) || REDUCTION_TYPES.includes(s));
};

function ancestorDependentSpecToIndependentSpec(
    spec: ViewSpec,
    ancestralFocus: AncestralFocus
) {
    const result = [...spec];
    for (let i = 0; i < result.length; i++) {
        const s = result[i];
        const matchingAncestor = ancestralFocus.filter((x) => x.key === s)[0];
        if (isNormalViewSpecDim(s)) {
            continue;
        } else if (matchingAncestor !== undefined) {
            result[i] = matchingAncestor.pick;
        } else {
            console.log(s, spec, ancestralFocus);
            throw new Error("cant resolve above spec");
        }
    }
    // console.log('Result of applying ancestral: ', { result, ancestralFocus, newSpec: spec });
    return result;
}

const defaultPanelUrlState = (lvnts: LazyVeryNamedTensor[], lvntIdx: number, options: any) => {
    const lvnt = lvnts[lvntIdx];
    const defaultViewSpec = (lvnt.dim_names.map(_ => "axis"));
    return {
        vizName: "Scalar",
        spec: defaultViewSpec,
        focus: lvnt.dim_idx_names.map(x => null),
        options: options,
        lvntIdx: lvntIdx,
        hover: lvnt.dim_names.map(_ => null),
    } as PanelUrlState;
};

export class TopLevel extends React.Component<TopLevelProps, TopLevelState> {
    numOutstandingRequests=0;
    hoverDebounceTimeoutId = -1;
    lastClearedTime=0;
    constructor(props: TopLevelProps) {
        super(props);
        const initialViewProps = this.props.initialViewProps;
        console.log("COMPOSABLE CONSTRUCTOR", { props: this.props.initialViewProps, urlState: this.props.urlState, state: this.state });
        this.state = {
            panels: [{ view: undefined, usedSpec: initialViewProps.lvnts[0].dim_names.map(_ => null), usedLvntIdx: 0 }],
            lvnts: initialViewProps.lvnts,
            nonce: -1,
            options:initialViewProps.options,
        };
        const newLvntDims = initialViewProps.lvnts.map(lvnt => lvnt.dim_idx_names.map(strings => strings.length));
        if (!this.props.urlState || JSON.stringify(this.props.urlState.lvntDims.map(x => x.length)) !== JSON.stringify(newLvntDims.map(x => x.length))) {
            this.updateUrlStateFixingFocusedDims({ lvntDims: newLvntDims, nonce: this.props.urlState ? this.props.urlState.nonce + 1 : 0, panels: [defaultPanelUrlState(initialViewProps.lvnts, 0, initialViewProps.options)] });
        } else {
            this.updateUrlStateFixingFocusedDims(this.props.urlState);
        }
        // Perform initial server call (if default spec was reduction, for example)
    }

    onDelete(viewIdx: number) {
        if (this.props.urlState.panels.length <= 1) return;
        let newPanels = [...this.props.urlState.panels];
        newPanels.splice(viewIdx, 1);
        this.updateUrlStateFixingFocusedDims({ ...this.props.urlState, panels: newPanels });
    }

    onDuplicate(viewIdx: number) {
        let ourState = this.props.urlState.panels[viewIdx];
        let newPanels = [...this.props.urlState.panels, {
            ...ourState,
            focus: getClearedFocus(ourState.focus),
            hover: ourState.hover.map(x => null),
        }];
        this.updateUrlStateFixingFocusedDims({ ...this.props.urlState, panels: newPanels });
    }

    onChangeSpec(viewIdx: number, change: ChangeRequest) {
        if (change.hover){
            if (change.hover.every(x=>x===null)){
                console.log("CLEARING")
                this.lastClearedTime=performance.now()
            }else if (this.lastClearedTime+1000>performance.now()){
                console.log("disregarding hover because of recent clear")
                return
            }
        }
        console.log("CHANGING")
        this._onChangeSpec(viewIdx, change);
    }

    getAncestorFocusKey(viewIdx: number, panelUrlState: PanelUrlState, dimIdx: number) {
        const lvnt = this.state.lvnts[panelUrlState.lvntIdx];
        return `${viewIdx}:${panelUrlState.vizName}[${lvnt.dim_names[dimIdx]}]`;
    }

    getAncestralFocus(urlState: ComposableUIUrlState | null = null) {
        if (urlState === null) urlState = this.props.urlState;

        let ancestralFocus = [] as AncestralFocus;
        for (let i = 0; i < urlState.panels.length; i++) {
            const props = urlState.panels[i];
            const lvnt = this.state.lvnts[props.lvntIdx];
            const theseResults = range(props.focus.length).map((j) => ({
                dim_type: lvnt.dim_types[j],
                dim_name: lvnt.dim_names[j],
                dim_idx_names: lvnt.dim_idx_names[j],
                key: this.getAncestorFocusKey(i, props, j),
                pick: props.hover[j] ?? props.focus[j] ?? props.spec[j],
                i: j,
                vizIdx: i
            }));
            ancestralFocus.push(...theseResults);
        }
        ancestralFocus = ancestralFocus.filter(x => isNormalViewSpecDim(x.pick));
        return ancestralFocus;
    }

    async _onChangeSpec(viewIdx: number, change: ChangeRequest) {
        if (change.focus && change.hover) {
            throw new Error('Should only change one of focus and hover at once!');
        }

        clearTimeout(this.hoverDebounceTimeoutId); // Prevent hovers from firing after change focus

        let newPanels = this.props.urlState.panels.map(x => ({ ...x }));
        const newPanel = newPanels[viewIdx];

        if (change.viewSpec) {
            if (newPanel.spec.some((x, i) => (x === "axis" || x === "facet") !== ((change.viewSpec as any)[i] === "axis" || (change.viewSpec as any)[i] === "facet"))) {
                newPanel.focus = change.viewSpec.map(x => null);
                newPanel.hover = change.viewSpec.map(x => null);
            }
            newPanel.spec = change.viewSpec;
        }
        if (change.focus) {
            newPanel.focus = change.focus;
            newPanel.hover = change.focus;
        }
        if (change.hover) {
            newPanel.hover = change.hover;
        }
        if (change.lvntIdx !== undefined) {
            const newLvnt = this.state.lvnts[change.lvntIdx];
            if (newPanel.spec.length !== newLvnt.dim_idx_names.length) {
                newPanel.spec = newLvnt.dim_idx_names.map(x => "axis");
                newPanel.focus = newLvnt.dim_idx_names.map(x => null);
                newPanel.hover = newLvnt.dim_idx_names.map(x => null);
            }
            newPanel.lvntIdx = change.lvntIdx;
        }
        if (change.vizName) {
            newPanel.vizName = change.vizName;
        }

        if (change.focus &&
            change.focus.some(x => x !== null) &&
            viewIdx === this.props.urlState.panels.length - 1) {

            const newPanelState = { ...newPanel };
            newPanelState.spec = newPanel.spec.map((s, i) => {
                if (newPanel.focus[i]!==null) {
                    return this.getAncestorFocusKey(viewIdx, newPanel, i);
                }
                return s;
            });

            newPanels.push(newPanelState);
        }
        this.updateUrlStateFixingFocusedDims({ ...this.props.urlState, panels: newPanels });
    }

    updateUrlStateFixingFocusedDims(unfinishedUrlState: ComposableUIUrlState) {

        const allValidAncestorKeys = unfinishedUrlState.panels.map((panelState, i) => panelState.spec.map((_, j) => this.getAncestorFocusKey(i, panelState, j))).flat();
        const allValidAncestorKeysObj = {} as { [k: string]: boolean; };
        for (let x of allValidAncestorKeys) {
            allValidAncestorKeysObj[x] = true;
        }
        const finishedUrlState = {
            ...unfinishedUrlState, panels: unfinishedUrlState.panels.map((panelState, panelIdx) => {
                const specDimInvalidMask = panelState.spec.map(x => {
                    if (typeof x === "string") {
                        return x[x.length - 1] === "]" && !allValidAncestorKeysObj[x as string]; //HACK
                    }
                    return false;
                });
                if (specDimInvalidMask.some(x => x)) {
                    panelState.focus = panelState.focus.map(x => null);
                    panelState.hover = panelState.hover.map(x => null);
                    panelState.spec = panelState.spec.map((s, i) => specDimInvalidMask[i] ? 'axis' : s);
                }
                return panelState;
            })
        };
        this.updateValidUrlState(finishedUrlState);
    }

    updateValidUrlState(urlState: ComposableUIUrlState) {

        const ancestralFocus = this.getAncestralFocus(urlState);
        const newBigPanels = urlState.panels.map((x, i) => this.state.panels[i] ? { ...this.state.panels[i] } : { view: undefined, usedSpec: [-1], usedLvntIdx: -1 } as PanelBigState);
        const newUsedSpecs = urlState.panels.map(urlPanelState => ancestorDependentSpecToIndependentSpec(urlPanelState.spec, ancestralFocus));
        const viewsToUpdate = range(newBigPanels.length).filter((i) => {
            const panelUrlState = urlState.panels[i];
            const bigState = newBigPanels[i];
            const spec = newUsedSpecs[i];
            if (!bigState) return true;
            return JSON.stringify(spec) !== JSON.stringify(bigState.usedSpec) || bigState.usedLvntIdx !== panelUrlState.lvntIdx;
        });
        const promises = viewsToUpdate.map(i => {
            return lvntGet(this.state.lvnts[urlState.panels[i].lvntIdx], newUsedSpecs[i]);
        });
        if (promises.length === 0 && newBigPanels.length===this.state.panels.length) {
            this.props.setUrlState(urlState);
        } else {
            if (this.numOutstandingRequests>0) {
                console.log("ignoring update while request is in flight");
                return;
            }
            console.log("Requesting");
            this.numOutstandingRequests +=1;
            Promise.all(promises).then(views => {
                views.forEach((view, viewToUpdateIdx) => {
                    const idx = viewsToUpdate[viewToUpdateIdx];
                    newBigPanels[idx].view = view;
                    newBigPanels[idx].usedLvntIdx = urlState.panels[idx].lvntIdx;
                    const urlPanelState = urlState.panels[idx];
                    newBigPanels[idx].usedSpec = newUsedSpecs[idx];
                });
                urlState.nonce += 1;
                const newState = { panels: newBigPanels, nonce: urlState.nonce };
                console.log("UPDATING COMPOSABLE STATE", { newState, urlState });
                this.numOutstandingRequests -=1;
                this.setState(newState);
                this.props.setUrlState(urlState);
            }).catch(()=>{
                console.log("PROMISE FAILED")
                this.numOutstandingRequests -=1;
            });
        }
    }


    render() {
        console.log("TopLevelRender", this.state, this.props.urlState);
        if (!this.props.urlState) {
            return (<p>no url state</p>);
        }
        if (!this.state) {
            return (<p>no big state</p>);
        }
        // if (this.props.urlState && this.props.urlState.nonce < this.state.nonce) {
        //     this.updateUrlStateFixingFocusedDims(this.props.urlState);
        //     return (<p>Fetching previous view</p>);
        // }

        if (this.props.urlState.nonce !== this.state.nonce) {
            console.log(`COMPOSABLE URL version ${this.props.urlState.nonce} STATE version ${this.state.nonce}`);
            return (<p>Loading from url</p>);
        }
        const ancestralFocus = this.getAncestralFocus();


        let views = this.state.panels.map((bigState, i) => {
            const urlState = this.props.urlState.panels[i];
            const handlers = {
                onChangeSpec: (change: ChangeRequest) => this.onChangeSpec(i, change),
                onDelete: () => this.onDelete(i),
                onDuplicate: () => this.onDuplicate(i)
            };
            return (<div style={{margin:"7px"}} key={i}>
                <div className="plot_box_title" title="which Tensor?">
                    Plot {i}, showing 
                    <select value={urlState.lvntIdx} style={{ width: "275px" }} onChange={(e) => {
                        this.onChangeSpec(i, { lvntIdx: parseInt(e.target.value) });
                    }} >
                        {this.state.lvnts.map((lvnt, n) => (<option key={n} value={n}>{lvnt.title}</option>))}
                    </select>
                    <button onClick={() => this.onDelete(i)}>Delete Plot</button>
                    <button onClick={() => this.onDuplicate(i)}>Duplicate Plot</button>
                </div>
                <View {...{ ...bigState, ...urlState, ...handlers, ancestralFocus: ancestralFocus.filter(x => x.vizIdx !== i), lvnt: this.state.lvnts[urlState.lvntIdx] }} unembedder={this.state.options.unembedder} />
            </div>);
        });
        // TODO: tooltip appropriateness - where should this properly go? 
        // {hoverDeets && hover.some(x => x !== null) && <NumTooltip value={hoverDeets[1]} units={lvnt.units} targetRef={hoverDeets[0]} />}

        return  <div className="plot_box">
                    {this.numOutstandingRequests>0 && <div style={{ backgroundColor: "yellow" }}> {this.numOutstandingRequests} Requests In Flight</div>}
                    <div className="section_title">
                        Composable Charts
                    </div>
                    {views.map(x=><> <div style={{border:"3px solid var(--border_purple)",height:"0px",width:"calc(100% + 1px)",borderWidth:"3px 0 0 0", margin:"0 -1px 0 -1px"}}></div>{x}</>).flat()}
                </div>;
    }
}