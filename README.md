# FSRC_BASE
## baseline

if mode == 'test':
   pseudo_label = torch.softmax(logitis, dim=-1)   #<100, 5>
   max_probs, max_idx = torch.max(pseudo_label, dim=-1) #<100>  <100>
   mask = max_probs.ge(0.85).float()
   query_protos=[[] for i in range(N)]
   for i in range(N*Q):
       if mask[i] == 1:
          query_protos[max_idx[i]].append(trans_query[i, :])
   for i in range(N):
       if len(query_protos[i]) >0:
          query_protos[i] = torch.cat([query_proto.unsqueeze(0) for query_proto in query_protos[i]])
          query_protos[i] = query_protos[i].mean(dim=0)
          proto[i, :] = 0.5 * proto[i, :] + 0.5 * query_protos[i]
   logitis = euclidean_metric(trans_query, proto) 
   logitis = logitis / self.cfg['temperature']   #<100, 5>
