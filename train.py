import argparse
import datetime

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

from utils.logger import get_root_logger
from utils.path import mkdir_or_exist

import warnings
warnings.filterwarnings("ignore")

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
	from apex import amp
except:
	print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
	mixed_precision = False  # not installed

# Hyperparameters
hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
	   'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
	   'momentum': 0.937,  # SGD momentum/Adam beta1
	   'weight_decay': 5e-4,  # optimizer weight decay
	   'giou': 0.05,  # giou loss gain
	   'cls': 0.5,  # cls loss gain
	   'cls_pw': 1.0,  # cls BCELoss positive_weight
	   'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
	   'obj_pw': 1.0,  # obj BCELoss positive_weight
	   'iou_t': 0.20,  # iou training threshold
	   'anchor_t': 4.0,  # anchor-multiple threshold
	   'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
	   'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
	   'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
	   'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
	   'degrees': 0.0,  # image rotation (+/- deg)
	   'translate': 0.0,  # image translation (+/- fraction)
	   'scale': 0.5,  # image scale (+/- gain)
	   'shear': 0.0}  # image shear (+/- deg)




def init_logger(work_dir='./work_dir'):
	cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	work_dir = os.path.join(work_dir, cur_time.split('_')[0], cur_time)

	mkdir_or_exist(os.path.abspath(work_dir))

	# log
	log_file = os.path.join(work_dir, 'log.log')
	logger = get_root_logger(log_file)
	return logger, work_dir


def load_yaml(yaml_file):
	file = open(yaml_file, 'r', encoding="utf-8")
	file_data = file.read()
	file.close()
	data = yaml.load(file_data)
	return data





def train(hyp, logger, work_dir, device):

	epochs = opt.epochs
	batch_size = opt.batch_size
	total_batch_size = opt.total_batch_size
	weights = opt.weights
	rank = opt.local_rank


	# Configure
	init_seeds(1)
	with open(opt.data) as f:
		data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
	#train_path = data_dict['train']
	#test_path = data_dict['val']
	train_path = os.path.join(data_dict['convertor_path'], 'images', 'train2017')
	test_path = os.path.join(data_dict['convertor_path'], 'images', 'val2017')
	nc, names = (1, ['item']) if opt.single_cls else (int(len(data_dict['names'])), data_dict['names'])  # number classes, names
	assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check


	# Create model
	model = Model(opt.cfg, nc=nc).to(device)

	# Image sizes
	gs = int(max(model.stride))  # grid size (max stride)
	imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

	# Optimizer
	nbs = 64  # nominal batch size
	# default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
	# all-reduce operation is carried out during loss.backward().
	# Thus, there would be redundant all-reduce communications in a accumulation procedure,
	# which means, the result is still right but the training speed gets slower.
	# TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
	# in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
	accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
	hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

	pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
	for k, v in model.named_parameters():
		if v.requires_grad:
			if '.bias' in k:
				pg2.append(v)  # biases
			elif '.weight' in k and '.bn' not in k:
				pg1.append(v)  # apply weight decay
			else:
				pg0.append(v)  # all else

	if hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
		optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
	else:
		optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

	optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
	optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
	logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
	del pg0, pg1, pg2

	# Load Model
	google_utils.attempt_download(weights)
	start_epoch, best_fitness = 0, 0.0

	# 加载自己的模型
	if not weights.endswith('.pt'):
		ckpt = torch.load(weights, map_location=device).float()
		model.load_state_dict(ckpt.state_dict(), strict=True)
		logger.info(f'load myself ckpt: {weights}')


	if weights.endswith('.pt'):  # pytorch format
		ckpt = torch.load(weights, map_location=device)  # load checkpoint

		# load model
		try:
			exclude = ['anchor']  # exclude keys
			ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
							 if k in model.state_dict() and not any(x in k for x in exclude)
							 and model.state_dict()[k].shape == v.shape}
			model.load_state_dict(ckpt['model'], strict=False)
			print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))
		except KeyError as e:
			s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
				"Please delete or update %s and try again, or use --weights '' to train from scratch." \
				% (weights, opt.cfg, weights, weights)
			raise KeyError(s) from e

		# load optimizer
		if ckpt['optimizer'] is not None:
			optimizer.load_state_dict(ckpt['optimizer'])
			best_fitness = ckpt['best_fitness']

		# load results
		if ckpt.get('training_results') is not None:
			with open(results_file, 'w') as file:
				file.write(ckpt['training_results'])  # write results.txt

		# epochs
		start_epoch = ckpt['epoch'] + 1
		if epochs < start_epoch:
			print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
				  (weights, ckpt['epoch'], epochs))
			epochs += ckpt['epoch']  # finetune additional epochs

		del ckpt




	# Mixed precision training https://github.com/NVIDIA/apex
	if mixed_precision:
		model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

	# Scheduler https://arxiv.org/pdf/1812.01187.pdf
	lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
	# https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
	# plot_lr_scheduler(optimizer, scheduler, epochs)

	# DP mode
	if device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	
	# SyncBatchNorm
	if opt.sync_bn and device.type != 'cpu' and rank != -1:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
		logger.info('Using SyncBatchNorm()')

	# Exponential moving average
	ema = torch_utils.ModelEMA(model) if rank in [-1, 0] else None

	# DDP mode
	if device.type != 'cpu' and rank != -1:
		model = DDP(model, device_ids=[rank], output_device=rank)

	# Trainloader
	dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
											cache=opt.cache_images, rect=opt.rect, local_rank=rank,
											world_size=opt.world_size)
	mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
	nb = len(dataloader)  # number of batches
	assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

	# Testloader
	if rank in [-1, 0]:
		# local_rank is set to -1. Because only the first process is expected to do evaluation.
		testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
									   cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]

	# Model parameters
	hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
	model.nc = nc  # attach number of classes to model
	model.hyp = hyp  # attach hyperparameters to model
	model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
	model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
	model.names = names

	# Class frequency
	if rank in [-1, 0]:
		labels = np.concatenate(dataset.labels, 0)
		c = torch.tensor(labels[:, 0])  # classes
		# cf = torch.bincount(c.long(), minlength=nc) + 1.
		# model._initialize_biases(cf.to(device))

		# Check anchors
		if not opt.noautoanchor:
			check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

		# save anchors
		m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
		anchors = []
		for i in range(3):
			for j in range(3):
				anchor = m.anchor_grid[i, 0, j, 0, 0].cpu().detach().numpy().tolist()
				anchors.append(anchor)
		with open(os.path.join(work_dir, 'anchors.txt'), 'w') as f:
			for anchor in anchors:
				f.write(f'{anchor[0]},{anchor[1]}\n')

	# Start training
	t0 = time.time()
	nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
	maps = np.zeros(nc)  # mAP per class
	results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
	scheduler.last_epoch = start_epoch - 1  # do not move
	if rank in [0, -1]:
		logger.info('Image sizes %g train, %g test' % (imgsz, imgsz_test))
		logger.info('Using %g dataloader workers' % dataloader.num_workers)
		logger.info('Starting training for %g epochs...' % epochs)
	# torch.autograd.set_detect_anomaly(True)
	for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
		train_time_start = time.time()
		logger.info('')
		logger.info('epoch: {epoch} lr: {lr}'.format(epoch=epoch, lr=optimizer.param_groups[0]['lr']))

		model.train()

		# Update image weights (optional)
		# When in DDP mode, the generated indices will be broadcasted to synchronize dataset.
		if dataset.image_weights:
			# Generate indices.
			if rank in [-1, 0]:
				w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
				image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
				dataset.indices = random.choices(range(dataset.n), weights=image_weights,
												 k=dataset.n)  # rand weighted idx
			# Broadcast.
			if rank != -1:
				indices = torch.zeros([dataset.n], dtype=torch.int)
				if rank == 0:
					indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
				dist.broadcast(indices, 0)
				if rank != 0:
					dataset.indices = indices.cpu().numpy()

		# Update mosaic border
		# b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
		# dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

		mloss = torch.zeros(4, device=device)  # mean losses
		if rank != -1:
			dataloader.sampler.set_epoch(epoch)
		'''
		pbar = enumerate(dataloader)
		if rank in [-1, 0]:
			logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
			pbar = tqdm(pbar, total=nb)  # progress bar
		'''
		optimizer.zero_grad()
		for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch -------------------------------------------------------------
			ni = i + nb * epoch  # number integrated batches (since train start)
			imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

			# Warmup
			if ni <= nw:
				xi = [0, nw]  # x interp
				# model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
				accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
				for j, x in enumerate(optimizer.param_groups):
					# bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
					x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
					if 'momentum' in x:
						x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

			# Multi-scale
			if opt.multi_scale:
				sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
				sf = sz / max(imgs.shape[2:])  # scale factor
				if sf != 1:
					ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
					imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

			# Forward
			pred = model(imgs)

			# Loss
			loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
			if rank != -1:
				loss *= opt.world_size  # gradient averaged between devices in DDP mode
			if not torch.isfinite(loss):
				logger.info('WARNING: non-finite loss, ending training ', loss_items)
				return results

			# Backward
			if mixed_precision:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			# Optimize
			if ni % accumulate == 0:
				optimizer.step()
				optimizer.zero_grad()
				if ema is not None:
					ema.update(model)


			if i % 200 == 0:
				logger.info('[Epoch:{epoch}/{epochs} iter:{iter}] loss:{loss}'.format(epoch=epoch, epochs=epochs-1, iter=i, loss=loss.item()))


			# end batch ------------------------------------------------------------------------------------------------

		# Scheduler
		scheduler.step()

		train_time_end = time.time()
		logger.info('train time: {train_time}s'.format(train_time=int(train_time_end-train_time_start)))


		# Only the first process in DDP mode is allowed to log or save checkpoints.
		if rank in [-1, 0]:
			# mAP
			if ema is not None:
				ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
			final_epoch = epoch + 1 == epochs
			if (epoch % data_dict['eval_interval'] == 0 and epoch != 0)or final_epoch:  # Calculate mAP
				results, maps, times = test.test(data_dict,
												 batch_size=total_batch_size,
												 imgsz=imgsz_test,
												 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
												 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
												 single_cls=opt.single_cls,
												 dataloader=testloader,
												 save_dir=work_dir)
				map50, map = results[2], results[3]
				logger.info(f'eval:   mAP@.5: {map50}    mAP@.5:.95: {map}')

				# 保存模型
				ckpt = ema.ema.module if hasattr(ema.ema, 'module') else ema.ema
				torch.save(ckpt, os.path.join(work_dir, 'epoch_{epoch}.pth'.format(epoch=epoch)))


		# end epoch ----------------------------------------------------------------------------------------------------
	# end training

	dist.destroy_process_group() if rank not in [-1, 0] else None
	torch.cuda.empty_cache()
	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')
	parser.add_argument('--data', type=str, default='config.yaml', help='data.yaml path')
	parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
	parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
	parser.add_argument('--rect', action='store_true', help='rectangular training')
	parser.add_argument('--resume', nargs='?', const='get_last', default=False,
						help='resume from given path/to/last.pt, or most recent run if blank.')
	parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
	parser.add_argument('--notest', action='store_true', help='only test final epoch')
	parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
	parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
	parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
	parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
	parser.add_argument('--weights', type=str, default='', help='initial weights path')
	parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
	parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
	parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
	parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
	opt = parser.parse_args()



	opt.weights = last if opt.resume else opt.weights
	opt.cfg = check_file(opt.cfg)  # check file
	opt.data = check_file(opt.data)  # check file
	opt.total_batch_size = opt.batch_size
	opt.world_size = 1

	data = load_yaml(opt.data)
	os.environ['CUDA_VISIBLE_DEVICES'] = data['gpu_ids']
	#opt.device = data['gpu_ids']

	opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
	device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
	if device.type == 'cpu':
		mixed_precision = False

	#opt.noautoanchor = True


	nc = len(data['names'])
	task_name = data['task_name']
	opt.imgsz = data['imgsz']
	opt.epochs = data['epochs']
	opt.batch_size = data['batch_size']
	opt.eval_interval = data['eval_interval']
	opt.weights = data['weights']




	# logger work_dir
	logger, work_dir = init_logger(work_dir=f'./work_dir/{task_name}') 
	
	wdir = work_dir + os.sep  # weights dir
	os.makedirs(wdir, exist_ok=True)
	last = wdir + 'last.pt'
	best = wdir + 'best.pt'
	results_file = 'results.txt'

	# print data opt
	logger.info('---------------------- data ----------------------')
	for key in data.keys():
		logger.info('{key}: {value}'.format(key=key, value=data[key]))
	logger.info('---------------------- data ----------------------')
	logger.info('')	

	# print cfg opt
	cfg = load_yaml(opt.cfg)
	logger.info('---------------------- cfg ----------------------')
	for key in cfg.keys():
		logger.info('{key}: {value}'.format(key=key, value=cfg[key]))
	logger.info('---------------------- cfg ----------------------')
	logger.info('')	
	
	


	# print hyp opt
	logger.info('---------------------- hyp ----------------------')
	for key in hyp.keys():
		logger.info('{key}: {value}'.format(key=key, value=hyp[key]))
	logger.info('---------------------- hyp ----------------------')
	logger.info('')	

	logger.info('---------------------- opt ----------------------')
	opt_dict = vars(opt)
	for key in opt_dict.keys():
		logger.info('{key}: {value}'.format(key=key, value=opt_dict[key]))
	logger.info('---------------------- opt ----------------------')
	logger.info('')

	# Train
	train(hyp, logger, work_dir, device)



