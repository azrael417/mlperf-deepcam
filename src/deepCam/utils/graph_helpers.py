import torch
import torch.distributed as dist

from .cuda_graph import capture_graph


def capture_model(pargs, net, input_shape, device, graph_stream=None, mode="module"):
    
    if mode == "module":
        # capture model by modules
    
        # xception layers
        train_example = [torch.ones( (pargs.local_batch_size, *input_shape), dtype=torch.float32, device=device)]

        # NHWC
        if pargs.enable_nhwc:
            train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        
        net.xception_features = capture_graph(net.xception_features, 
                                              tuple(t.clone() for t in train_example), 
                                              graph_stream = graph_stream,
                                              warmup_iters = 10,
                                              use_amp = pargs.enable_amp and not pargs.enable_jit)
                                              
        ## bottleneck
        #train_example = [torch.ones( (pargs.local_batch_size, 2048, 48, 72), 
        #                              dtype=torch.float32,
        #                              device=device)]
        #
        #if pargs.enable_nhwc:
        #    train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        #
        #net.bottleneck = capture_graph(net.bottleneck,
        #                               tuple(t.clone().requires_grad_() for t in train_example),
        #                               graph_stream = graph_stream,
        #                               warmup_iters = 10,
        #                               use_amp = pargs.enable_amp and not pargs.enable_jit)
        
        ## upsample
        #train_example = [torch.ones( (pargs.local_batch_size, 256,  48,  72), dtype=torch.float32, device=device),
        #                 torch.ones( (pargs.local_batch_size,  48, 192, 288), dtype=torch.float32, device=device)]
        #
        #if pargs.enable_nhwc:
        #    train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        #
        #net.upsample = capture_graph(net.upsample, 
        #                             tuple(t.clone().requires_grad_() for t in train_example), 
        #                             graph_stream = graph_stream,
        #                             warmup_iters = 10,
        #                             use_amp = pargs.enable_amp and not pargs.enable_jit)
                                        
    elif mode == "model":
        
        train_example = [torch.ones( (pargs.local_batch_size, *input_shape), dtype=torch.float32, device=device)]
        
        # NHWC
        if pargs.enable_nhwc:
            train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
            
        net = capture_graph(net, 
                            tuple(t.clone() for t in train_example), 
                            graph_stream = graph_stream,
                            warmup_iters = 10,
                            use_amp = pargs.enable_amp and not pargs.enable_jit)
    
    # capture graph
    #net_train.module = cg.capture_graph(net_train.module, 
    #                                    tuple(t.clone() for t in train_example), 
    #                                    graph_stream = scaffolding_stream,
    #                                    warmup_iters = 10,
    #                                    use_amp = pargs.enable_amp and not pargs.enable_jit)
    
    #net_train = cg.capture_graph(net_train, 
    #                             tuple(t.clone() for t in train_example), 
    #                             graph_stream = scaffolding_stream,
    #                             warmup_iters = 10,
    #                             use_amp = pargs.enable_amp and not pargs.enable_jit)
    
    return net
    
    
