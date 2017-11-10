let seed: int = [%bs.raw "parseInt(Math.random() * Number.MAX_SAFE_INTEGER)"];
Random.init(seed);

type transferFunc = {
  func: (float) => float,
  deriv: (float) => float
};

type input = Perceptron(perceptron) | Input(float)
and outputs = Perceptrons(array(perceptron)) | Output(float)
and perceptron = {
  mutable inputNodes: array(input),
  mutable outputNodes: outputs,
  mutable weights: array(float),
  mutable output: float,
  mutable invalid: bool,
  mutable dEdN: float,
  transfer: transferFunc,
};

type networkInputs = array(input);

let makeWeights = fun(length: int) => {
  Array.map(
    fun(_) => Random.float(1.0),
    Array.make(length, 0.0));
};

let relu = {
  func: fun(inp: float) => {
    log(1.0 +. exp(inp));
  },
  deriv: fun(inp: float) => {
    1.0 /. (1.0 +. exp(-. inp));
  }
};

let sum = fun(arr: array(float)) => {
  Array.fold_right(
    (x, y) => x +. y,
    arr,
    0.0
  );
};

let rec calcOutput = fun(perc: perceptron) => {
  switch perc.invalid {
    /* perceptron value needs to be recalculated */
    | true => {
      /* multiply inputNode values by weights and pass through transfer function */
      perc.output = perc.transfer.func(Array.fold_right(
        (x, y) => x +. y,
        Array.mapi(
          (ind, inputNode) => {
            switch inputNode {
              | Perceptron(thisPerc) => calcOutput(thisPerc) *. Array.get(perc.weights, ind)
              | Input(inp) => inp *. Array.get(perc.weights, ind);
            }
          },
          perc.inputNodes
        ),
        0.0
      ));
      /* Revalidate output */
      perc.invalid = false;
      /* If this is an output node, update the outputNode value */
      ignore(switch perc.outputNodes {
        | Output(_) => perc.outputNodes = Output(perc.output)
        | Perceptrons(_) => ();
      });
    
      perc.output;
    }
    | false => perc.output
  }
};

let makePerceptron = fun(inputs: array(input)) => {
  let perc = {
    /* Add an input of 1.0 for bias */
    inputNodes: Array.append(inputs, [|Input(1.0)|]),
    outputNodes: Output(0.0),
    /* An extra weight for the bias */
    weights: makeWeights(Array.length(inputs) + 1),
    output: 0.0,
    dEdN: 0.0,
    invalid: true,
    transfer: relu
  };

  /* Update any input nodes which are perceptrons to have the new node as an output */
  ignore(Array.iter(
    (thisPerc) => {
      switch thisPerc {
        | Perceptron(thisThisPerc) => {
          thisThisPerc.outputNodes = switch thisThisPerc.outputNodes {
            | Perceptrons(percs) => Perceptrons(Array.append(percs, [|perc|]))
            | Output(_) => Perceptrons([|perc|])
          }
        }
        | Input(_) => ();
      }
    },
    inputs
  ));

  perc;
};

type network = array(array(perceptron));

let rec recursivelyMakeNetwork = fun(shape: list(int), net: option(network)) => {
  let inputs = switch net {
    | None => {
      Array.make(List.hd(shape), Input(0.0));
    }
    | Some(net) => {
      Array.map(
        (perc) => Perceptron(perc),
        Array.get(net, Array.length(net) - 1)
      );
    }
  };

  let existing = switch net {
    | None => [||]
    | Some(net) => net
  };

  switch shape {
    | [] => {
      existing;
    }
    | [layerSize, ...layers] => {
      recursivelyMakeNetwork(
        layers,
        Some(Array.append(
          existing,
          [|Array.map(
            (_) => makePerceptron(inputs),
            Array.make(layerSize, 0)
          )|]
        ))
      )
    }
  }
};

let makeNetwork = fun(shape: list(int)) => {
  recursivelyMakeNetwork(shape, None);
};

let calcOutput = fun(net: network) => {
  let outputLayer = Array.get(net, Array.length(net) - 1);
  Array.map(
    (perc) => calcOutput(perc),
    outputLayer
  );
};

let setInputs = fun(net: network, inputs: networkInputs) => {
  let inputLayer = Array.get(net, 0);
  ignore(Array.iter(
    (perc) => {
      perc.inputNodes = inputs;
      /* Check if all inputs are floats */
      /* If so, reduce the node weight array length to not include bias */
      /* Otherwise warn that connecting more perceptrons this way is weird */
      if (Array.fold_right(
        (input, allFloats) => switch input {
          | Perceptron(_) => false;
          | Input(_) => allFloats
        },
        inputs,
        true
      )) {
        ignore(Array.map(
          (perc) => perc.weights = Array.sub(perc.weights, 0, Array.length(inputs)),
          inputLayer
        ));
      } else {
        Js.log("WARNING - you are connecting perceptrons rather than values as network inputs");
      };
    },
    inputLayer
  ));
  ignore(Array.iter(
    (layer) => Array.iter(
      (perc) => perc.invalid = true,
      layer
    ),
    net
  ));

  net;
};

exception WrongNodeType(string);

let calcError = fun(net: network, expectedOutput: array(outputs)) => {
  let outputs = calcOutput(net);
  0.5 *. sum(Array.mapi(
    (ind, thisOutput) => {
      switch thisOutput {
        | Output(x) => (x -. Array.get(outputs, ind)) ** 2.0
        | Perceptrons(_) => raise(WrongNodeType("Perceptron"));
      }
    },
    expectedOutput
  ));
};

let logOutput= fun(net: network) {
  Js.log("Inputs");
  ignore(Array.iter(
    (inputNode) => switch inputNode {
      | Perceptron(_) => Js.log("Perceptron")
      | Input(inp) => Js.log(inp)
    },
    Array.get(Array.get(net, 0), 0).inputNodes
  ));
  Array.mapi(
    (ind, layer) => {
      Js.log("Layer " ++ string_of_int(ind));
      Array.map(
        (perc) => Js.log(perc.output),
        layer
      )
    },
    net
  );
};

let logWeights = fun(net: network) {
  ignore(Array.mapi(
    (ind, layer) => {
      Js.log("Layer " ++ string_of_int(ind));
      Array.map(
        (perc) => Js.log(perc.weights),
        layer
      )
    },
    net
  ));
};

type linearObj = Vector(array(float)) | Matrix(array(linearObj));
exception DimensionMismatch(string);

let rec linearAdd = fun(a: linearObj, b: linearObj): linearObj {
  switch a {
    | Vector(u) => {
      switch b {
        | Vector(v) => {
          Vector(Array.mapi(
            (ind, entry) => entry +. Array.get(u, ind),
            v
          ));
        }
        | Matrix(_) => {
          raise(DimensionMismatch("Objects do not have the same dimensions"));
        }
      }
    }
    | Matrix(m) => {
      switch b {
        | Vector(_) => {
          raise(DimensionMismatch("Objects do not have the same dimensions"));
        }
        | Matrix(n) => {
          Matrix(Array.mapi(
            (ind, entry) => linearAdd(entry, Array.get(n, ind)),
            m
          ));
        }
      }
    }
  };
};

let sumDerivs = fun(derivs: list(linearObj)) {
  List.fold_right(
    (thisError, errorSum) => linearAdd(thisError, errorSum),
    List.tl(derivs),
    List.hd(derivs)
  )
};

let errorDerivs = fun(net: network, expectedOutput: array(outputs)) {
  let reversed = Array.of_list(List.rev(Array.to_list(net)));
  Matrix(Array.of_list(List.rev(Array.to_list(Array.map(
    (layer) => {
      Matrix(Array.mapi(
        (ind, perc) => {
          let dEdO = switch perc.outputNodes {
            | Output(output) => {
              let thisExpectedOutput = Array.get(expectedOutput, ind);
              switch thisExpectedOutput {
                | Output(floatOutput) => output -. floatOutput
                | Perceptrons(_) => raise(WrongNodeType("Perceptron"))
              };
            }
            | Perceptrons(perceptrons) => {
              sum(Array.mapi(
                (innerInd, innerPerc) => {
                  innerPerc.dEdN *. Array.get(innerPerc.weights, innerInd)
                },
                perceptrons
              ));
            }
          };
          let dOdN = perc.transfer.deriv(perc.output);
          perc.dEdN = dEdO *. dOdN;
          Vector(Array.map(
            (input) => {
              perc.dEdN *. switch input {
              | Perceptron(innerPerc) => innerPerc.output
              | Input(float) => float
            };
          },
          perc.inputNodes
        ));
      },
      layer
      ));
    },
    reversed
  )))));
};

let updateWeights = fun(net: network, errorDerivs: linearObj, alpha: float) {
  switch errorDerivs {
    | Matrix(layers) => {
      ignore(Array.mapi(
        (indA, layer) => {
          switch layer {
            | Matrix(percEntries) => {
              ignore(Array.mapi(
                (indB, percEntry) => {
                  switch percEntry {
                    | Vector(weightErrors) => {
                      ignore(Array.mapi(
                        (indC, weightError) => {
                          let perc = Array.get(Array.get(net, indA), indB);
                          let currentWeight = Array.get(perc.weights, indC);
                          Array.set(perc.weights, indC, currentWeight -. (alpha *. weightError));
                          switch perc.outputNodes {
                            | Perceptrons(outputPercs) => {
                              ignore(Array.map(
                                (outputPerc) => outputPerc.invalid = true,
                                outputPercs
                              ))
                            }
                            | Output(_) => ()
                          }
                        },
                        weightErrors
                      ));
                    }
                    | _ => raise(DimensionMismatch("Object does not have the right number of dimensions"));
                  }
                },
                percEntries
              ))
            }
            | _ => raise(DimensionMismatch("Object does not have the right number of dimensions"));
          }
        },
        layers
      ))
    }
    | _ => raise(DimensionMismatch("Object does not have the right number of dimensions"));
  }
};

type trainingData = list((array(input), array(outputs)));

let sample = fun(min: float, max: float) {
  Random.float(max -. min) +. min;
};

let makeSingleValFunctionData = fun(f: (float) => float, min: float, max: float, number: int): trainingData {
  let examples = ref([]);
  for (_ in 0 to number) {
    let input = sample(min, max);
    examples := [([| Input(input) |], [| Output(f(input)) |]), ...examples^];
  };
  examples^;
};

let trainExample = fun(net: network, errorData: option((linearObj, float)), example: (array(input), array(outputs))) {
  let (inputs, outputs) = example;
  ignore(setInputs(net, inputs));
  ignore(calcOutput(net));
  let error = calcError(net, outputs);
  let derivs = errorDerivs(net, outputs);
  switch errorData {
    | None => (derivs, error)
    | Some(errorDataSoFar) => {
      let (weightErrorsSoFar, errorSum) = errorDataSoFar;
      (linearAdd(weightErrorsSoFar, derivs), error +. errorSum)
    }
  };
};

let trainEpoch = fun(net: network, data: trainingData, alpha: float) {
  let errorData = List.fold_left(
    (errorSoFar, example) => Some(trainExample(net, errorSoFar, example)),
    None,
    data
  );
  switch errorData {
    | None => 0.0
    | Some(errorData) => {
      let (weightErrors, errorSum) = errorData;
      ignore(updateWeights(net, weightErrors, alpha));
      errorSum;
    }
  }
};

let myNet = makeNetwork([5, 5, 1]);
let data = makeSingleValFunctionData((inp) => Pervasives.abs_float(Pervasives.sin(inp)), -5.0, 5.0, 100);
Js.log(trainEpoch(myNet, data, 0.001));
Js.log(trainEpoch(myNet, data, 0.001));
Js.log(trainEpoch(myNet, data, 0.001));
Js.log(trainEpoch(myNet, data, 0.001));
Js.log(trainEpoch(myNet, data, 0.001));
