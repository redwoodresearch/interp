import React, { useEffect, useState } from 'react';
import ReactDOM, { unmountComponentAtNode } from 'react-dom';
import * as msgpack from '@msgpack/msgpack';
import ndarray from 'ndarray';

import { View } from "./View";
import { TopLevel } from './TopLevel';
import { ViewRegistry } from './ViewRegistry';
import { AttribSetSpec, AttributionBackend, AttributionStateSpec, ComposableUIUrlState, InterpsiteUrlState, LazyVeryNamedTensor, QueriedLogits } from './proto';
import { isDeepSubtree, range } from './common';
import { AttributionUI } from './AttributionUI';
import {
  HashRouter as Router,
  Route,
  Routes,
  Link,
  useParams,
  useLocation,
  useNavigate,
  useSearchParams,
  Navigate,
} from "react-router-dom";

import {
  Float16Array, isFloat16Array, isTypedArray,
  getFloat16, setFloat16,
  hfround,
} from "@petamoriken/float16";
import { AttributionUISet } from './AttributionUISet';

const TEXTAREA_HEIGHT = "125px";
const TEXTAREA_WIDTH = "700px";

function rrinterpMsgpackDecode(
  buf: any,
  callbackChannel: (id: string, host: string | null, data: any) => Promise<any>,
): any {
  let result: any = msgpack.decode(buf);

  function walkTree(x: any): any {
    if (x === null || typeof x !== 'object')
      return x;
    if (x['$$type$$'] !== undefined) {
      switch (x['$$type$$']) {
        case 'ndarray':
          const slice = x.data.buffer.slice(x.data.byteOffset, x.data.byteLength + x.data.byteOffset);
          let array: ndarray.TypedArray;
          switch (x.dtype) {
            case 'int8': array = new Int8Array(slice); break;
            case 'uint8': array = new Uint8Array(slice); break;
            case 'int16': array = new Int16Array(slice); break;
            case 'uint16': array = new Uint16Array(slice); break;
            case 'int32': array = new Int32Array(slice); break;
            case 'uint32': array = new Uint32Array(slice); break;
            case 'int64': array = new Int32Array(slice); break; // Intentional mismatch!
            case 'uint64': array = new Uint32Array(slice); break; // Intentional mismatch!
            case 'float16': array = (() => {
              let stime = performance.now();
              const f16 = new Float16Array(slice);
              console.log("f16 took", performance.now() - stime); stime = performance.now(); const f32 = new Float32Array(f16); console.log("f32 took", performance.now() - stime); return f32;
            })(); break;
            case 'float32': array = new Float32Array(slice); break;
            case 'float64': array = new Float64Array(slice); break;
            default:
              alert('Bad serialized ndarray dtype: ' + x.dtype);
              return null;
          }
          return ndarray(array, x.shape);
        case 'torch':
          return walkTree(x.array);
        case 'jax':
          return walkTree(x.array);
        case 'local-callback':
          return (...args: any) => callbackChannel(x.id, null, args);
        default:
          alert('Bad serialized $$type$$: ' + x['$$type$$']);
          return null;
      }
    }
    if (Array.isArray(x))
      return x.map(walkTree);
    if (x instanceof Uint8Array)
      return x;

    const result: any = {};
    for (const [k, v] of Object.entries(x))
      result[k] = walkTree(v);
    return result;
  }
  result = walkTree(result);
  console.log("got msgpack", result);

  return result;
}

export class Comms {
  websocket: any;
  callbackTokenCounter = 0;
  callbackPromiseResolveTable = new Map<number, (data: any) => void>();
  onopens: (() => any)[] = [];
  port: string;
  url: string;
  onopen() {
    this.send({ kind: "init" });
    this.onopens.forEach(x => x());
    this.onopens = [];
  }
  handlers = {} as { [key: string]: (message: any) => void; };
  constructor(port: string, url: string) {
    window.onbeforeunload = () => {
      this.websocket?.close();
    };
    this.port = port;
    this.url = url;
    this.resetWebsocket();
  }

  resetWebsocket() {
    if (this.websocket) {
      this.websocket.close();
    }
    this.websocket = new WebSocket(`ws://${this.url}:${this.port}/`);
    // causes onMessage's ev.data to be an ArrayBuffer instead of Blob.
    this.websocket.binaryType = "arraybuffer";
    this.websocket.onopen = this.onopen.bind(this);
    this.websocket.onmessage = this.onMessage.bind(this);
  }

  send(data: any) {
    const jsonData = JSON.stringify(data);
    const sendStuff = () => {
      this.websocket.send(jsonData);
    };
    if (this.websocket.readyState !== WebSocket.OPEN) {
      this.onopens.push(sendStuff);
      this.resetWebsocket();
    } else {
      sendStuff();
    }
  }

  onMessage = (ev: any) => {
    let message = rrinterpMsgpackDecode(ev.data, this.callbackChannel);
    // CM: special case View and TopLevel because they are not in the viewRegistry
    if (message.kind === "callbackResult") {
      if (message.buffer === "unknown callback token") {
        window.location.reload();
        return;
      }
      const resolve = this.callbackPromiseResolveTable.get(message.token);
      if (resolve === undefined) {
        alert('Unknown callbackResult token!');
      } else {
        resolve(message.buffer);
      }
    } else if (this.handlers[message.kind]) {
      this.handlers[message.kind](message);
    } else {
      console.log("Unrecognized message type: " + message.kind);
    }
  };

  callbackChannel = (id: string, host: string | null, data: any): Promise<any> => {
    this.callbackTokenCounter++;
    return new Promise((resolve) => {
      this.callbackPromiseResolveTable.set(this.callbackTokenCounter, resolve);
      console.log({ Sending: "Sending" }, data);
      let message = {
        kind: 'callback',
        id,
        token: this.callbackTokenCounter,
        data
      };
      this.send(message);
    });
  };

  isConnectedOrConnecting() {
    return (!!this.websocket && (this.websocket.readyState === WebSocket.OPEN || this.websocket.readyState === WebSocket.CONNECTING));
  }

  isConnected() {
    return (!!this.websocket && this.websocket.readyState === WebSocket.OPEN);
  }

  addOnOpen(x: () => void) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      x();
    } else {
      this.onopens.push(x);
    }
  }

  addHandler(kind: string, fn: (x: any) => void) {
    this.handlers[kind] = fn;
  }
}

console.log("Starting up standalone page controller!");

interface ModelAbout {
  name: string;
  info: {
    model_config: {
      num_layers: number;
      num_heads: number;
      hidden_size: number;
      norm_type: string;
      pos_enc_type: string;
    };
    model_class: string;
  };
}

const defaultStrings = {
  paulMcCartney: `The musician named Paul McCartney wrote Here Comes The Sun. Paul McCartney is in the band The Beatles.`, eiffelTower: `The landmark Eiffel Tower, is surrounded by beautiful Paresian shops.`, manySentences: `Here are some example sentences:
Amy walked her dog around the park, and frolicked in the grass.
Bob wore his brown shirt every day, and loved how he looked in it.
Chloe played MTG every day at seven pm.
Danny plays baseball with his father in the field.`};

let thang = [] as Partial<InterpsiteUrlState>[];
let interpsiteNonce = 0;
function TextEntryAndLVNTSelection(props: any) {
  const { comms }: { comms: Comms; } = props;
  let defaultText = `Welcome to the Redwood Interpretability Website! Click "How to Use" for documentation. If you have feedback, add comments to the doc or contact tao at rdwrs dot com. The Redwood Interpretability team wishes you the best in your interpretability endeavors!`;
  const defaultBigState = { makers: { lvntMakers: [] as { name: string; fn: (text: string, name: string) => Promise<LazyVeryNamedTensor>; required_model_info_subtree: any; }[], backendMaker: null as null | { fn: (s: string, name: string) => Promise<AttributionBackend>; }, availableModels: [] as ModelAbout[], unembedder: null as null | {fn:((vector: ndarray.NdArray,modelName:string) => Promise<QueriedLogits>)} }, activeLVNTs: [] as LazyVeryNamedTensor[], attributionBackend: null as null | AttributionBackend, nonce: -1 };

  const defaultUrlState = { whichModel: 0, prompt: defaultText, nonce: 0, allNonce: 0, whichAttributionUI: "tree" } as InterpsiteUrlState;
  let urlState: InterpsiteUrlState = defaultUrlState;
  if (props.searchParamsAndSet[0].get("interpsite")) {
    urlState = JSON.parse(props.searchParamsAndSet[0].get("interpsite"));
  }
  const [sideBySide, setSideBySide] = useState(false);
  const [showComposable, setShowComposable] = useState(true);
  const [showAttribution, setShowAttribution] = useState(true);
  const [promptInProgress, setPromptInProgress] = useState(urlState.prompt);
  const [state, setState] = useState(defaultBigState);


  const justSetUrlState = (newUrlState: InterpsiteUrlState) => {
    newUrlState.allNonce = interpsiteNonce + 1;
    interpsiteNonce += 1;
    let updatedSearchParams = new URLSearchParams(props.searchParamsAndSet[0].toString());
    updatedSearchParams.set("interpsite", JSON.stringify(newUrlState));
    props.searchParamsAndSet[1](updatedSearchParams.toString());
    // console.log("pushing history", decodeURIComponent(updatedSearchParams.toString()));
  };

  const processUrlStateUpdateBacklog = () => {
    if (thang.length > 0 && urlState.allNonce === interpsiteNonce) {
      let obj = { ...urlState };
      thang.forEach(x => {
        obj = { ...obj, ...x };
      });
      justSetUrlState(obj);
      thang = [];
    }
  };

  const justSetUrlStatePartial = (newUrlState: Partial<InterpsiteUrlState>) => {
    thang.push(newUrlState);
    if (urlState.allNonce === interpsiteNonce) {
      processUrlStateUpdateBacklog();
    }
  };

  const updateUrlState = (newUrlState: InterpsiteUrlState) => {
    const promptBeginIfNeeded = newUrlState.prompt;
    newUrlState.nonce += 1;
    const promises = getModelReleventLVNTMakers(newUrlState).map(x => x.fn(promptBeginIfNeeded, state.makers.availableModels[newUrlState.whichModel].name));
    promises.push((state.makers.backendMaker as any).fn(promptBeginIfNeeded, state.makers.availableModels[newUrlState.whichModel].name));
    Promise.all(promises).then((lvnts) => {
      console.log("lvnts", lvnts);
      const ab = lvnts.pop() as any as AttributionBackend;
      justSetUrlState(newUrlState);
      setState({ ...state, attributionBackend: ab, activeLVNTs: lvnts, nonce: newUrlState.nonce });
    });
  };

  const getModelReleventLVNTMakers = (urlState: InterpsiteUrlState) => {
    return state.makers.lvntMakers.filter((maker, i) => {
      const subobj = maker.required_model_info_subtree;
      return isDeepSubtree(state.makers.availableModels[urlState.whichModel].info, subobj);
    });
  };


  const onConnectCallback = () => {
    comms.send({ "kind": "getTensorMakers" });
  };
  const onAvailableLVNTs = (message: any) => {
    setState({ ...state, makers: { lvntMakers: message.availableLVNTMakers, backendMaker: { fn: message.attributionBackend }, availableModels: message.availableModels, unembedder: {fn:message.unembedder} } });
  };

  useEffect(() => {
    comms.addHandler("availableLVNTMakers", onAvailableLVNTs);
    comms.addOnOpen(onConnectCallback);
  }, []);

  useEffect(() => {
    if (state.makers.backendMaker !== null || state.makers.lvntMakers.length > 0 && state.nonce === -1)
      updateUrlState({ ...urlState });
  }, [state.makers]);

  // if (urlState.nonce < state.nonce) {
  //   updateUrlState({ ...urlState });
  // }

  if (urlState.nonce !== state.nonce) {
    console.log(`URL version ${urlState.nonce} STATE version ${state.nonce}`);
    return (<p>Loading from url</p>);
  }
  const modelDetailsToShow: any = { ...state.makers.availableModels[urlState.whichModel].info.model_config };
  delete modelDetailsToShow.attn_probs_dropout_rate;
  delete modelDetailsToShow.dropout_rate;
  delete modelDetailsToShow.embed_dropout_rate;
  delete modelDetailsToShow.layer_norm_epsilon;
  delete modelDetailsToShow.max_sequence_len;

  processUrlStateUpdateBacklog();
  return (
    <div>
      <div id="title">
        Transformer Visualizer
      </div>
      <div style={{ display: "flex", flexDirection: "row" }} className="top_links">
        <div><a href="https://docs.google.com/document/d/1ECwTXrgTqgiMN24L7IantJTaFpyJM2LxXXGq50meFKc/edit" target="_blank" className="top_link">How to Use</a></div>
        <div><a href="https://docs.google.com/document/d/1BEFD80M3wxQrV_L3RjIk-H8TwyWpUQPIdKL6Hv8Duqk/edit" target="_blank" className="top_link">Share your Results</a></div>
        <div><a href="https://docs.google.com/document/TEMPTEMPTEMP" target="_blank" className="top_link">Give Feedback</a></div>
      </div>


      <div className="box_container">
        <div className="box" style={{ display: "flex", flexDirection: "row", width: "100%" }}>
          <div>
            <div style={{ display: "flex", flexDirection: "row" }}><div style={{ margin: "7px 2px 0 0" }}> Running model </div><select value={urlState.whichModel} onChange={(e) => updateUrlState({ ...urlState, whichModel: parseInt(e.target.value) })}>{state.makers.availableModels.map((m, i) => <option key={m.name} label={m.name} value={i}></option>)}</select> <div style={{ margin: "7px 2px 0 2px" }}><a href="https://docs.google.com/document/d/1ECwTXrgTqgiMN24L7IantJTaFpyJM2LxXXGq50meFKc/edit#heading=h.w8buhx31beyl" target="_blank">About this model</a></div></div>

            <div style={{ margin: "6px 0 11px 8px" }}> on the text </div>
            {!comms.isConnected() && <div style={{ backgroundColor: "red", margin: "5px", padding: "5px" }}>Not Connected</div>}
            <div style={{ display: "flex", flexDirection: "row" }}>
              <textarea value={promptInProgress} onChange={(e) => setPromptInProgress(e.target.value)} style={{ width: TEXTAREA_WIDTH, height: TEXTAREA_HEIGHT, fontSize: "16px", padding: "6px", borderRadius: "6px", margin: "0 3px 4px" }} title="Ctrl-enter to update text" onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.shiftKey || e.metaKey)) {
                  updateUrlState({ ...urlState, prompt: promptInProgress });
                  e.preventDefault();
                }
              }} />
            </div>
            <button className="button" onClick={() => updateUrlState({ ...urlState, prompt: promptInProgress })} disabled={urlState.prompt === promptInProgress} style={{ backgroundColor: urlState.prompt !== promptInProgress ? "var(--encouraged)" : "", margin: "5px" }}>Update Text</button>
          </div>
          <div style={{ flexGrow: "1" }}></div>
          <div style={{ marginLeft: "7px" }}>
            <table>

              <div className="box_title"> Display options: </div>
              <tr>
                <td>
                  Show composable charts: &nbsp;
                </td>
                <td>
                  <label className="switch">
                    <input type="checkbox" checked={showComposable} onClick={() => setShowComposable(!showComposable)} />
                    <span className="slider round"></span>
                  </label>
                </td>
              </tr>

              <tr>
                <td>Show attribution graph: &nbsp; </td>
                <td>
                  <label className="switch">
                    <input type="checkbox" checked={showAttribution} onClick={() => setShowAttribution(!showAttribution)} />
                    <span className="slider round"></span>
                  </label>
                </td>
              </tr>


              <tr>
                <td>Show stacked: &nbsp;</td>
                <td>
                  <label className="switch" >
                    <input type="checkbox" onClick={() => setSideBySide(!sideBySide)} />
                    <span hidden={!showComposable || !showAttribution} className="slider round"></span>
                  </label>
                </td>
              </tr>

            </table>
          </div>
        </div>
      </div>



      <div {...(sideBySide && { style: { display: "flex", flexDirection: "row" } }) as any}>
        <div id="composable">
          <TopLevel setUrlState={(x: ComposableUIUrlState) => {
            console.log("UPDATING URL STATE", x);
            justSetUrlState({ ...urlState, composableUI: x });
          }} urlState={urlState.composableUI as ComposableUIUrlState} initialViewProps={{ options: {unembedder:(v:ndarray.NdArray)=>state.makers.unembedder?.fn(v,state.makers.availableModels[urlState.whichModel].name)}, lvnts: state.activeLVNTs }} />
        </div>
        <div id="attribution">
          <div style={{ margin: "0 16px" }}> Attribution Mode:
            <button onClick={() => updateUrlState({ ...urlState, whichAttributionUI: (urlState.whichAttributionUI === "tree" ? "set" : "tree"), attributionUI: undefined })} style={{ backgroundColor: urlState.whichAttributionUI === "tree" ? "var(--pressed)" : "white" }}> Tree</button>
            <button onClick={() => updateUrlState({ ...urlState, whichAttributionUI: (urlState.whichAttributionUI === "tree" ? "set" : "tree"), attributionUI: undefined })} style={{ backgroundColor: urlState.whichAttributionUI === "set" ? "var(--pressed)" : "white" }}> Set</button>
          </div>
          {state.attributionBackend && (urlState.whichAttributionUI === 'tree' ? (<AttributionUI setUrlState={(x: AttributionStateSpec) => justSetUrlState({ ...urlState, attributionUI: x })} urlState={urlState.attributionUI as AttributionStateSpec} attributionBackend={state.attributionBackend} options={{}} />) :
            (<AttributionUISet setUrlState={(x: AttribSetSpec) => justSetUrlState({ ...urlState, attributionUI: x })} urlState={urlState.attributionUI as AttribSetSpec} attributionBackend={state.attributionBackend} options={{}} />))}</div>
      </div>
    </div>
  );
};

function TextEntry(props: any) {
  const { comms, name, defaultString }: { comms: Comms; name: string; defaultString: string; } = props;
  const [promptInProgress, setPromptInProgress] = useState(defaultString ?? `[BEGIN] Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr Dursley was the director of a firm called Grunnings, which made drills.`);
  const [lvntsMaker, setLvntsMaker] = useState(null as null | { fn: (s: string) => Promise<LazyVeryNamedTensor[]>; });
  const [lvnts, setLvnts] = useState([] as LazyVeryNamedTensor[]);
  const onConnectCallback = () => {
    comms.send({ "kind": "nameStartup", "name": name });
  };
  const onStartup = (message: any) => {
    setLvntsMaker(message.data);
  };
  useEffect(() => {
    comms.addHandler("nameStartup", onStartup);
    comms.addOnOpen(onConnectCallback);
  }, []);

  const getLVNTs = () => {
    if (lvntsMaker) {
      lvntsMaker.fn(promptInProgress).then((lvnts) => {
        setLvnts(lvnts);
      });
    }
  };
  return (
    <div>
      <textarea value={promptInProgress} onChange={(e) => setPromptInProgress(e.target.value)} style={{ width: TEXTAREA_WIDTH, height: TEXTAREA_HEIGHT }} onKeyDown={(e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.shiftKey)) {
          getLVNTs();
        }
      }} />
      <div><button className="button" onClick={getLVNTs}>Submit</button></div>
      <div>
        {lvnts.length > 0 &&
          <p>BROKEN IN REFACTOR, ASK TAO TO FIX</p>
          // <TopLevel key={lvnts.map(x => `${x.title}${x.units}`).join("-")} initialViewProps={{ options: {}, lvnts: lvnts }}  />
        }
      </div>
    </div>
  );
};

function JustTensors({ comms, name, searchParamsAndSet }: { comms: Comms; name: string; searchParamsAndSet: any; }) {
  const [lvnts, setLvnts] = useState([] as LazyVeryNamedTensor[]);
  const onConnectCallback = () => {
    comms.send({ "kind": "nameStartup", "name": name });
  };
  const onStartup = (message: any) => {
    console.log("setting lvnts", message.data);
    setLvnts(message.data);
  };
  useEffect(() => {
    comms.addHandler("nameStartup", onStartup);
    comms.addOnOpen(onConnectCallback);
  }, []);

  let urlState = { composableUI: null as unknown as ComposableUIUrlState };
  if (searchParamsAndSet[0].get("justtensors")) {
    urlState = JSON.parse(searchParamsAndSet[0].get("justtensors"));
  }
  const setUrl = (newUrlState: any) => {
    let updatedSearchParams = new URLSearchParams(searchParamsAndSet[0].toString());
    updatedSearchParams.set("justtensors", JSON.stringify({ "composableUI": newUrlState }));
    searchParamsAndSet[1](updatedSearchParams.toString());
  };

  return (
    <div>
      {lvnts.length > 0 &&
        <TopLevel key={lvnts.map(x => `${x.title}${x.units}`).join("-")} initialViewProps={{ options: {}, lvnts: lvnts }} urlState={urlState.composableUI} setUrlState={setUrl} />
      }
    </div>
  );
};

function Communicator({ element }: any) {
  const urlParams = useParams();
  const { port, url } = urlParams;
  const [comms, setComms] = useState(() => new Comms(port ?? process.env.REACT_APP_MY_PORT ?? "6789", url ?? process.env.REACT_APP_MY_IP ?? "127.0.0.1"));
  const Component = element;
  const searchParamsAndSet = useSearchParams();
  return <div>
    <Component comms={comms} {...urlParams} searchParamsAndSet={searchParamsAndSet} />
  </div>;
}

function App(props: any) {
  return (<ErrorBoundary>
    <Router>
      <Routes>
        <Route path="/tensors/:name" element={<Communicator element={JustTensors} />} />
        <Route path="/functions/:name" element={<Communicator element={TextEntry} />} />
        {/* <Route path="/attribution/:name" element={<Communicator element={JustAttribution} />} /> */}
        <Route path="/" element={<Communicator element={TextEntryAndLVNTSelection}></Communicator>} />
        <Route path="/starter" element={<Navigate replace to="/?attribution=%7B%22tree%22%3A%5B%7B%22idx%22%3A%5B2%2C1%2C45%2C0%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B2%2C1%2C45%2C1%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B2%2C1%2C45%2C2%5D%2C%22children%22%3A%5B%7B%22idx%22%3A%5B1%2C4%2C45%2C0%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B1%2C4%2C45%2C1%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B1%2C4%2C45%2C2%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B1%2C4%2C45%2C0%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B1%2C4%2C45%2C1%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B1%2C4%2C45%2C2%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B2%2C6%2C45%2C0%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B2%2C6%2C45%2C1%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%2C%7B%22idx%22%3A%5B2%2C6%2C45%2C2%5D%2C%22children%22%3A%5B%5D%2C%22threshold%22%3A0.1%7D%5D%2C%22root%22%3A%7B%22kind%22%3A%22logprob%22%2C%22data%22%3A%7B%22seqIdx%22%3A45%2C%22tokString%22%3A%22+mysterious%22%2C%22comparisonTokString%22%3Anull%7D%2C%22threshold%22%3A0.1%7D%2C%22lineWidthScale%22%3A1%2C%22useIGAttn%22%3Afalse%2C%22useIGOutput%22%3Afalse%2C%22threshold%22%3A0.1%2C%22specificLogits%22%3A%5B%5D%7D&composable=%255B%257B%2522spec%2522%253A%255B%2522mean%2522%252C%2522mean%2522%252C%2522axis%2522%252C%2522axis%2522%255D%252C%2522options%2522%253A%257B%257D%252C%2522vizName%2522%253A%2522TextAverageTo%2522%252C%2522focus%2522%253A%255Bnull%252Cnull%252C75%252Cnull%255D%252C%2522lvntIdx%2522%253A1%257D%252C%257B%2522spec%2522%253A%255B%2522mean%2522%252C%2522mean%2522%252C%25220%253ATextAverageTo%255BQ%255D%2522%252C%2522axis%2522%255D%252C%2522options%2522%253A%257B%257D%252C%2522vizName%2522%253A%2522ColoredText%2522%252C%2522focus%2522%253A%255Bnull%252Cnull%252Cnull%252C48%255D%252C%2522lvntIdx%2522%253A1%257D%252C%257B%2522spec%2522%253A%255B%2522axis%2522%252C%2522axis%2522%252C%25220%253ATextAverageTo%255BQ%255D%2522%252C%25221%253AColoredText%255BK%255D%2522%255D%252C%2522options%2522%253A%257B%257D%252C%2522vizName%2522%253A%2522SidedMatrix%2522%252C%2522focus%2522%253A%255B1%252C1%252Cnull%252Cnull%255D%252C%2522lvntIdx%2522%253A1%257D%252C%257B%2522spec%2522%253A%255B%2522axis%2522%252C%2522max%2522%255D%252C%2522options%2522%253A%257B%257D%252C%2522vizName%2522%253A%2522ColoredText%2522%252C%2522focus%2522%253A%255Bnull%252C1%255D%252C%2522lvntIdx%2522%253A3%257D%252C%257B%2522spec%2522%253A%255B%2522axis%2522%252C%2522max%2522%255D%252C%2522options%2522%253A%257B%257D%252C%2522vizName%2522%253A%2522ColoredText%2522%252C%2522focus%2522%253A%255B46%252Cnull%255D%252C%2522lvntIdx%2522%253A3%257D%252C%257B%2522spec%2522%253A%255B%25224%253AColoredText%255Bseq%255D%2522%252C%2522axis%2522%255D%252C%2522options%2522%253A%257B%257D%252C%2522vizName%2522%253A%2522Tops%2522%252C%2522focus%2522%253A%255Bnull%252Cnull%255D%252C%2522lvntIdx%2522%253A3%257D%255D" />}></Route>
      </Routes>
    </Router>
  </ErrorBoundary>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: any) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true };
  }

  render() {
    if ((this.state as any).hasError) {
      // You can render any custom fallback UI
      return <h1>If the rest of the page is blank, this URL is likely corrupted / made with an old server version. Go to root site
        <a href="http://interp-tools.redwoodresearch.org">interp-tools.redwoodresearch.org</a> and navigate back to the vizes you want</h1>;
    }

    return this.props.children;
  }
}

ReactDOM.render(React.createElement(App),
  document.getElementById("widgetsmain"));