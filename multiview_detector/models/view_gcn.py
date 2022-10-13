import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tokenize import Double
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import os
import glob
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name


    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(), 
                os.path.join(complete_path, 
                    "model-{}.pth".format(str(epoch).zfill(5))))


    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].to(torch.float32) #!
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id

class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        vertices = vertices.to(torch.float32) #!
        id = knn(self.k, vertices, vertices)
        F = index_points(F,id)
        v = index_points(vertices,id)
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        v_F = v_F.to(torch.float32)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        F = torch.sum(F,-2)
        return F

class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, 40*self.s_views))
    def forward(self,F,vertices,k):
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,40).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new

class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self,F,V):
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)
        return F

class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F


mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))

class view_GCN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='resnet18', num_views=20):
        super(view_GCN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        self.vertices = torch.tensor(vertices).cuda()

        self.LocalGCN1 = LocalGCN(k=4,n_views=self.num_views)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views//2)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views//2)
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views//4)
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2)
        self.View_selector2 = View_selector(n_views=self.num_views//2, sampled_view=self.num_views//4)

        self.cls = nn.Sequential(
            nn.Linear(512*3,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        views = self.num_views
        y = self.net_1(x)
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)

        y = self.LocalGCN1(y,vertices)
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]

        z, F_score, vertices2 = self.View_selector1(y2,vertices,k=4)
        z = self.LocalGCN2(z,vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]

        w, F_score2, vertices3 = self.View_selector2(z2,vertices2,k=4)
        w = self.LocalGCN3(w,vertices3)
        pooled_view3 = torch.max(w, 1)[0]

        pooled_view = torch.cat((pooled_view1,pooled_view2,pooled_view3),1)
        pooled_view = self.cls(pooled_view)
        return pooled_view,F_score,F_score2
