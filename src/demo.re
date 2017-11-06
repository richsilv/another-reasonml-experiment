type input = Perceptron(perceptron) | Input(float)
and outputs = Perceptrons(array(perceptron)) | Output(float)
and perceptron = {
  mutable inputNodes: array(input),
  mutable outputNodes: outputs,
  mutable weights: array(float),
  mutable output: float,
  mutable invalid: bool,
  transfer: (float) => float,
};

type networkInputs = array(input);

let makeWeights = fun(length: int) => {
  Array.map(
    fun(_) => Random.float(1.0),
    Array.make(length, 0.0));
};

let relu = fun(inp: float) => {
  log(1.0 +. exp(inp));
};

let rec calcOutput = fun(perc: perceptron) => {
  switch perc.invalid {
    /* perceptron value needs to be recalculated */
    | true => {
      /* multiply inputNode values by weights and pass through transfer function */
      perc.output = perc.transfer(Array.fold_right(
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
    inputNodes: inputs,
    outputNodes: Output(0.0),
    weights: makeWeights(Array.length(inputs)),
    output: 0.0,
    invalid: false,
    transfer: relu
  };

  /* Update any input nodes which are perceptrons to have the new node as an output */
  ignore(Array.map(
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

let rec makeNetworkRecursive = fun(shape: list(int), net: option(network)) => {
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
      makeNetworkRecursive(
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
  makeNetworkRecursive(shape, None);
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
  ignore(Array.map(
    (perc) => perc.inputNodes = inputs,
    inputLayer
  ));
  ignore(Array.map(
    (layer) => Array.map(
      (perc) => perc.invalid = true,
      layer
    ),
    net
  ));

  net;
};

let logOutput= fun(net: network) {
  Js.log("Inputs");
  ignore(Array.map(
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

let z = makeNetwork([2, 3, 2]);
setInputs(z, [|Input(0.4), Input(0.2)|]);

calcOutput(z);
logOutput(z);
