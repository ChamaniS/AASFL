# training arguments

import argparse
pl = 0.7
pl_1=0.5
npl=0.0
def args_parser():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--image_height', type=int, default=256,
                        help="image_height")
    parser.add_argument('--image_width', type=int, default=256,
                        help="image_width")

    # federated arguments
    parser.add_argument('--rounds', type=int, default=10,
                        help="number of global rounds of training for both R and NR clients")
    parser.add_argument('--comrounds', type=int, default=10,
                        help="number of global rounds of training for both R and NR clients")
    parser.add_argument('--roundsR', type=int, default=3,
                        help="number of global rounds of training for R clients")
    parser.add_argument('--roundsNR', type=int, default=1,
                        help="number of global rounds of training for NR clients")
    parser.add_argument('--num_users', type=int, default=2,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=12,
                        help="the number of local epochs for R clients: E")
    parser.add_argument('--Rlocal_ep', type=int, default=12,
                        help="the number of local epochs for R clients: E")
    parser.add_argument('--NRlocal_ep', type=int, default=12,
                        help="the number of local epochs for NR clients: E")
    parser.add_argument('--screen_ep', type=int, default=1,
                        help="the number of screening epochs: E")
    parser.add_argument('--val_global_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1,
                        help="local batch size: B")
    parser.add_argument('--fixed_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_R', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_NR', type=float, default=0.00000555555,
                        help='learning rate')
    parser.add_argument('--mode', type = str, default='fedprox',
                        help='fedavg | fedprox | fedbn')
    parser.add_argument('--mupure', type=float, default=0.01,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--mubad', type=float, default=0.07,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--pack_prob_up_C1', type=float, default=npl,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C1', type=float, default=npl,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C2', type=float, default=npl,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C2', type=float, default=npl,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C3', type=float, default=pl,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C3', type=float, default=pl,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C4', type=float, default=pl,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C4', type=float, default=pl,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--pack_prob_up_C5', type=float, default=pl,
                        help='Packet loss probability: Uplink')
    parser.add_argument('--pack_prob_down_C5', type=float, default=pl,
                        help='Packet loss probability: Downlink')
    parser.add_argument('--max_retra_deep', type=float, default=10,
                        help='Maximum no.of retransmission attempts in the deep split')
    parser.add_argument('--max_retra_shallow', type=float, default=10,
                        help='Maximum no.of retransmission attempts in the shallow split')

    # other arguments
    parser.add_argument('--num_classes', type=int, default=5, help="number \
                        of classes")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
