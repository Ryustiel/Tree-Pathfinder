from binarytree import Node, tree, bst
from math import sqrt
from random import randint
from time import time
import replit

global parameters
parameters = [1, 0.5, 0, 1, 0, 2, 0] #[egalite, loss, scoref, arrondi, zero]
def setParameters(win=False, egalite=False, loss=False, scoref=False, mxscoref=False, arrondi=False, zero=False):
  global parameters
  i = 0
  for parameter in (win, egalite, loss, scoref, mxscoref, arrondi, zero):
    if parameter is not False:
      parameters[i] = parameter
    i = i + 1

class aNode:
  def __init__(self, value=0, left=0, right=0):
    self.value = value
    self.left = left
    self.right = right
  def __str__(self):
    return "\ntableau <===============>\n\n" + parcours(self) + "\n<===============>\n"

def parcours(node):
  f = []
  rstring = ""
  f.insert(0, node)

  linebreak = 0 #nombre d'elements de l'arbre avant saut de ligne
  i = 1

  while (len(f) != 0):
    node = f.pop()
    rstring = rstring + str(node.value) #ajoute valeur

    if i >= 2**linebreak:
      rstring = rstring + "\n"
      linebreak = linebreak + 1 #puissance de 2 suivante (étage suivant)
    else:
      i = i + 1

    if node.left != 0:
      f.insert(0, node.left)
    elif node.value != (-1, "edge "):
      f.insert(0, aNode((-1, "edge "))) #limites
    if node.right != 0:
      f.insert(0, node.right)
    elif node.value != (-1, "edge "):
      f.insert(0, aNode((-1, "edge ")))

  return rstring

def tab2tree(tab, k1=0, k2=0):
  """
  transforme un tableau en arbre personnalisé (pouvant contenir des tuple)
  """
  try:
    tab[k1][k2]
  except:
    return aNode((0, "empty"))
  return aNode(tab[k1][k2], tab2tree(tab, k1+1, k2), tab2tree(tab, k1, k2+1))



# FONCTIONS SERIEUSES ====================================================

def tdepth(lroot, currdepth=0): #currdepth =/= currendepth mais == max depth dans branches below
  """
  retourne la profondeur de l'arbre
  """
  if lroot is None:
    return currdepth

  else: 
    currdepth = currdepth + 1
    return max(tdepth(lroot.left, currdepth), tdepth(lroot.right, currdepth))

def rowtrunc(tab):
  """
  A DEBUGGER
  return le tableau avec la colonne de gauche en moins 
  """
  result = []
  for line in tab:
    result.append(line[1:]) #ERREUR INTERNE POTENTIELLE A GERER : LE TABLEAU N'A PAS D'INDICE [1:] (mais la fonction marcha quand meme)
  return result

def diagtrunc(tab, inverted=False):
  """
  A DEBUGGER (voir rowtrunc) VOIR CETTE FONCTION DEJA DEBUGGEE LEN(DIAG) > 1
  opérations sur les tableaux de diagonales...
  """
  result = []

  for diagonale in tab:
    if inverted:
      result.append(diagonale[:-1])
    else:
      result.append(diagonale[1:])


  return result


#FONCTIONS DE TRAITEMENT STATISTIQUE =========================================

def moyenne_diagonale(liste):
  """
  fonction spéciale adaptée à la configuration des tableaux de diagonales
  """
  m = 0
  i = 0
  for value in liste:
    if value is not None: #ignore les fillers
      m = m + value
      i = i + 1

  if i == 0:
    return None
  else:
    return m / i

def moyenne(liste):
  m = 0
  for value in liste:
    m = m + value
  return m / len(liste)

def ecart(liste, moyenne):
  e = 0
  for value in liste:
    e = e + (value - moyenne)**2
  return sqrt( e /len(liste) )

def lmax(liste):
  mx = liste[0]
  for item in liste:
    if item > mx:
      mx = item
  return mx

def lmin(liste):
  mn = liste[0]
  for item in liste:
    if item < mn:
      mn = item
  return mn

def flatten(liste, mx, mn, arrondi):
  assert mx != mn
  flatliste = []
  topcap = mx - mn #valeur proportionnelle supérieure (topcap [0, 1] vaut 1 proportionnellement)
  for item in liste:
    flatliste.append(round( (item-mn) / topcap, arrondi)) #loi de proportionnalité
  return flatliste

#FONCTIONS DE MODELISATION ===============================================================

def tab2btree(tab, profondeur=-1):
  """
  transforme un tableau en arbre binaire donc chaque branche correspond à un déroulement possible de la partie (on peut aller sois dans la case de droite, sois dans la case d'en bas sur le tableau)

  profondeur <= -1 : l'arbre est généré à partir de l'entièreté des lignes suivantes
  """
  if (profondeur == 0): #profondeur maximale atteinte
    return None
  else:
    try:
      tab[0][0]
    except:
      return None

    return Node(tab[0][0], tab2btree(tab[1:], profondeur-1), tab2btree(rowtrunc(tab), profondeur-1))
  

def simplified_tree(tab, ct=0):
  """
  tab de diagonales
  depuis une diagonale partielle : on prend la moyenne des valeurs de cette diagonale partielle comme valeur, on prend sois la diagonale partielle suivante comme référence, sois on garde deux niveaux de flou.

  """
  try:
    tab[-1][0]
  except: #si tab vide noeud none
    return None

  value = moyenne_diagonale(tab[0])

  if value is None: #si il n'y a pas de valeur (segment de diagonale hors du tableau)
    return None

  if ct > 0:
    return Node(value, simplified_tree(diagtrunc(tab[1:], True), ct - 1), simplified_tree(diagtrunc(tab[1:]), ct - 1)) #diagonales suivantes tronquées d'une unité (sans la ligne supérieure ou la ligne inférieure des noeuds possibles à partir des noeuds représentés par la moyenne)
    
  else:
    return Node(value, simplified_tree(tab[1:], ct)) #on passe à la diagonale suivante


def diagonals(tab):
  """
  tableau de diagonales de tab
  """
  result = []
  #diagonales partant de la gauche
  for j in range(len(tab) + len(tab[0]) - 1): #colonne + ligne (- intersection colonne ligne)
    result.append([]) #nouveau tableau
    for i in range(len(tab[0]) + len(tab)):
      jj = j - i

      if jj >= 0:
        try:
          value = tab[jj][i]
        except:
          result[-1].append(None) #valeurs virtuelles
        else:
          result[-1].append(value)
      else:
        break

  return result


def sumto(lnode, depth=0, sum=0):
  """
  ajoute la valeur des noeuds plus hauts au noeud à la profondeur spécifiée, et ce pour chaque noeud à cette profondeur.
  """

  #VALUES
  if (depth == 0):
    val = lnode.value + sum
  else:
    val = lnode.value

  #TESTS
  r = (lnode.right is None)
  l = (lnode.left is None)

  if l and r: #inscrit la somme générée au cours de la récursion si les deux S.A. sont vides
    return Node(val)

  elif not l and not r:
    return Node(val, sumto(lnode.left, depth-1, sum+lnode.value), sumto(lnode.right, depth-1, sum+lnode.value))

  elif not l and r:
    return Node(val, sumto(lnode.left, depth-1, sum+lnode.value), None)

  else:
    return Node(val, None, sumto(lnode.right, depth-1, sum+lnode.value))


def sumit(lroot):
  for i in range(tdepth(lroot), 0, -1): #summation
    lroot = sumto(lroot, i)
  return lroot

def treeturn(lroot, bin=False):
  if lroot is None:
    return
  if bin:
    return Node(lroot.value*-1, treeturn(lroot.left, False), treeturn(lroot.right, False))
  else:
    return Node(lroot.value, treeturn(lroot.left, True), treeturn(lroot.right, True))


def BDtab_copy(tab):
  """
  copie un tableau bidimensionnel (éviter l'écriture globale)
  """
  result = []
  for j in range(len(tab)):
    result.append([])
    for i in range(len(tab[0])):
      result[-1].append(tab[j][i])

  return result


def TDtab_copy(tab):
  """
  copie un tableau tridimensionnel (éviter l'écriture globale)
  """
  result = []
  for i in range(len(tab)):
    result.append(BDtab_copy(tab[i]))

  return result


def turn(tab, bin=False):
  linebin = not bin #négation sur la prochaine ligne

  for j in range(len(tab)):
    for i in range(len(tab[0])):
      if bin:
        tab[j][i] = -1 * tab[j][i]
      bin = not bin #alterne entre signe et pas signe sur la ligne
    bin = linebin
    linebin = not linebin #début de la ligne suivante inversé (début d'une diagonale sur deux)
  return tab


#FONCTIONS D'ANALYSE ===========================================================

#REPARTITION DES VALEURS
#...dans le tableau
def grid_to_line(tab):
  result = []
  
  for line in tab:
    for value in line:
      result.append(value)

  return result
    

#ISSUES

def brscore(lroot):
  """
  score maximum parmi les issues de l'arbre (int)
  """
  r = (lroot.right is None)
  l = (lroot.left is None)

  if r and l:
    return lroot.value
  elif not r and not l:
    return max(brscore(lroot.left), brscore(lroot.right))
  elif not r and l:
    return brscore(lroot.right)
  else:
    return brscore(lroot.left)


def scores(lroot): #tableau de tous les scores
  """
  retourne un tableau des scores des issues de l'arbre [int, int, ...]
  """
  r = (lroot.right is None)
  l = (lroot.left is None)

  if (r and l):
    return (lroot.value, )
  elif not r and not l:
    liste = []
    for value in scores(lroot.left):
      liste.append(value)
    for value in scores(lroot.right):
      liste.append(value)
    return liste
  elif not r and l:
    liste = []
    for value in scores(lroot.right):
      liste.append(value)
    return liste
  else:
    liste = []
    for value in scores(lroot.left):
      liste.append(value)
    return liste

def winstats(lroot, zero=0):
  """
  nombre de victoires, égalités, défaites (avec x score de l'issue : resp x>0, x==0, x<0)\n
  retourne un tuple d'int (victoires, égalités, défaites)
  """
  r = (lroot.right is None)
  l = (lroot.left is None)

  if (r and l):
    if lroot.value > zero:
      return (1, 0, 0)
    elif lroot.value == zero:
      return (0, 1, 0)
    else:
      return (0, 0, 1)

  elif not r and not l:
    r = winstats(lroot.right, zero)
    l = winstats(lroot.left, zero)
    return (r[0]+l[0], r[1]+l[1], r[2]+l[2]) #tous les succes de la branche

  elif not r and l:
    return winstats(lroot.right, zero)
  else:
    return winstats(lroot.left, zero)

#FONCTIONS D'EVALUATION =========================================

def odds(lroot):
  """
  depth Win
  depth Gain (pck les calculs de gain prennent plus de temps)
  """
  global parameters
  mxscoref = parameters[4]

  mx = lmax(scores(lroot))
  mn = lmin(scores(lroot))

  if (lroot.left is None) or (lroot.right is None):
    isgreater = 0 #si il n'y a pas deux sous arbres, chercher de quel côté est le maximum n'a pas d'intérêt
    mxscoref = 0
  else:
    isgreater = (brscore(lroot.left) > brscore(lroot.right)) #le facteur n'est appliqué qu'à gauche ou droite

  return (
    (winrate((lroot.left), mx, mn) + isgreater * mxscoref)    /(1 + isgreater * mxscoref), 
    (winrate((lroot.right), mx, mn) + (not isgreater) * mxscoref)   /(1 + (not isgreater) * mxscoref)
    )


def winrate(lroot, mx, mn):

  if (lroot is None): #le chemin n'est pas empruntable
    return -1
  else:
    
    if mx == mn: #toutes les valeurs pour ce sous arbre et l'autre sous arbre (deux exécutions distinctes de cette fonction) sont les mêmes; pas d'intérêt de comparer quoi que ce soit
      if mn >= 0: #on compare la valeur mn; arbitraire (mn = mx)
        return 1 #on est certain de gagner
      else:
        return 0 #impossible de gagner

    else:
      global parameters
      winf = parameters[0]
      egalite = parameters[1]
      loss = parameters[2]
      scoref = parameters[3]
      arrondi = parameters[5]
      zero = parameters[6]

      wins = winstats(lroot, zero)
      gains = scores(lroot)
      m = moyenne(gains)
      e = ecart(gains, m)

      mayscale = flatten((m+e,), mx, mn, arrondi)[0] #indicateur d'espérance de bon score

      issues = 0
      for i in range(3):
        issues = issues + wins[i]

      winrate = (winf*wins[0] + egalite*wins[1] + loss*wins[2]) / issues
      winrate = winrate + scoref * mayscale

      return winrate/ (1 + scoref) #pour rester entre 0 et 1


#PLATO DE JEU =====================================================

def random_tab(i=None, j=None):
  if i is None:
    i = randint(2, 8)
  if j is None:
    j = randint(2, 8)
  tab = []
  for line_id in range(j):
    tab.append([])
    for val_id in range(i):
      tab[line_id].append(randint(0, 10))

  return tab

def game_format(tab):
  format_tab = []
  i = 0
  j = 0
  for j in range(len(tab)):
    format_tab.append([])
    for i in range(len(tab[0])):
      format_tab[j].append( [tab[j][i], 0] )

  return format_tab

def game_display(tab):
  result = "" #string contenant l'affichage final du tableau de jeu
  linebreak = ""

  for i in range(len(tab[0])): #ligne de saut de ligne
    linebreak = linebreak + "-----" #la longueur de la ligne dépend du nombre de colonnes et tab[0] est la première ligne quelconque d'un tableau dont toutes les lignes ont le même nombre d'éléments
  linebreak = linebreak + "\n"

  for line in tab:
    result = result + linebreak #ajoute le saut de ligne entre chaque ligne
    
    for (cell, attribute) in line:

      cell = str(cell)
      for i in range(3 - len(cell)):
        cell = " " + cell

      if attribute == 0:
        cell = '\033[35m\033[49m' + cell #mauve
      elif attribute == 1:
        cell = '\033[1m\033[32m\033[49m' + cell #vert
      elif attribute == 2:
        cell = '\033[1m\033[94m\033[49m' + cell #bleu
      elif attribute == 3:
        cell = '\033[1m\033[31m\033[49m' + cell #rouge

      result = result + "\033[0m\033[39m|" + cell #ajoute une case au rendu final

    result = result + "\033[0m\033[39m\033[49m|\n"

  return result


#JEU =====================================

class GameInstance:
  """
  INSTANCE DE JEU
  """
  def __init__(self, grid, iaTurn=False):
    """
    iaTurn Trie : l'ia joue en deuxième, c'est à dire que c'est la première à choisir parmis deux possibilités.
    """
    self.game_tab = game_format(grid)
    self.grid = turn(BDtab_copy(grid), iaTurn) #grille de jeu ajustée (gain indépendant de chaque case)
    self.iaTurn = iaTurn
    self.i = 0 #coordonnées de la case (0; 0) de grid, dans game_tab
    self.j = 0

    #gestion des scores
    self.iaScore = 0 #le score d'ia est recalculé systématiquement à partir de la case actuelle du chemin
    if iaTurn :
      self.playerScore = self.game_tab[0][0][0] #l'ia choisit en premier = elle ne gagne pas la valeur de la toute première case
      self.game_tab[0][0][1] = 2 #coloration joueur (case de départ jouée)
    else:
      self.playerScore = 0 #score du joueur spécifique au joueur...
      self.game_tab[0][0][1] = 3 #coloration ia

  def isDone(self):
    """
    True si plateau de jeu parcouru
    """
    if len(self.grid) == 1 and len(self.grid[0]) == 1:
      return True
    else:
      return False

  def playerInput(self, bot):
    if not self.iaTurn and not self.isDone():

      self.iaScore = self.iaScore + self.grid[0][0] #met à jour le score de l'ia (case avant déplacement)

      if bot:
        self.grid = self.grid[1:] #retire la ligne du haut
        self.j = self.j + 1
        self.game_tab[self.j][self.i][1] = 2 #coloration joueur
      else:
        self.grid = rowtrunc(self.grid)
        self.i = self.i + 1
        self.game_tab[self.j][self.i][1] = 2

      self.playerScore = self.playerScore + self.game_tab[self.j][self.i][0] #valeur de la case atteinte à l'instant (pas mis à jour par la fonction iaPlay)

      self.iaTurn = True #au tour de l'IA

  def iaPlay(self):
    if self.iaTurn and not self.isDone():

      setParameters(zero = self.iaScore) #définit le zero pour le winstats

      side_winrate = odds(sumit(simplified_tree(diagonals(self.grid), 3)))
      bot = (side_winrate[0] >= side_winrate[1]) #NIVEAU DE SIMPLIFICATION

      self.iaScore = self.iaScore + self.grid[0][0] #met à jour le score de l'ia (case avant déplacement)

      if bot:
        self.grid = self.grid[1:] #retire la ligne du haut
        self.j = self.j + 1
        self.game_tab[self.j][self.i][1] = 3 #coloration ia
      else:
        self.grid = rowtrunc(self.grid)
        self.i = self.i + 1
        self.game_tab[self.j][self.i][1] = 3

      self.iaTurn = False #au tour du Player

  def simulate_path(self):
    """
    simule le reste de la partie
    """
    instance = GameInstance(turn(BDtab_copy(self.grid), not self.iaTurn), True) #tableau actuel
    instance.iaScore = self.iaScore
    instance.game_tab = TDtab_copy(self.game_tab)
    instance.i = self.i
    instance.j = self.j

    while not instance.isDone():
      instance.iaPlay() 
      instance.iaTurn = True #réinitialise le tour pour l'ia (elle joue pour elle même)
      instance.game_tab[instance.j][instance.i][1] = 1 #change la couleur pour du vert

    return instance

  def __str__(self):
    return game_display(self.game_tab)


#=====================================

def live(instance):

  while not instance.isDone():
    
    if instance.iaTurn:
      instance.iaPlay()

    else:
      replit.clear()
      print(instance.simulate_path())
      print(instance.playerScore, "VS", instance.iaScore + instance.playerScore + instance.grid[0][0]) #grid 0; 0 = case actuelle de l'ia pas comptée
      
      p = input("\ns ou d\n=> ")
      if p == "s":
        p = True
      else:
        p = False
      
      instance.playerInput(p)

  replit.clear()
  print(instance.simulate_path())
  print(instance.playerScore, "VS", instance.iaScore + instance.playerScore)
  print("FINI")



#TESTING ==========================================================================
def issues(lroot):
  if lroot is None:
    return
  elif lroot.left is None and lroot.right is None:
    print(lroot.value, end=" | ")
  else:
    issues(lroot.left)
    issues(lroot.right)

def nbissues(lroot):
  if lroot is None:
    return 0
  elif lroot.left is None and lroot.right is None:
    return 1
  else:
    return nbissues(lroot.left) +nbissues(lroot.right)


#VRAI TAB DE TESTS =======

  
#TEST D'AFFICHAGE ET DE TRUCS ========/================/===========================

def plan_de_simplification(tab):
  """
  tableau de simplifications en fonction de la longueur du tableau et de la position des extremums...
  """
  print(tab)

ttab = random_tab(20, 20)
wuw = diagonals(turn(BDtab_copy(ttab)))

print(game_display(game_format(ttab)))

print(issues(sumit(simplified_tree(wuw, 0))))
print(issues(sumit(simplified_tree(wuw, 2))))
print(issues(sumit(simplified_tree(wuw, 4))))
print(issues(sumit(simplified_tree(wuw, 6))))

timers = []
r = [99999]
r = [0]
for i in range(0, 7, 2):
  r.insert(-1, i)

for i in r:
  curr = time()
  h = sumit(simplified_tree(wuw, i))
  curr = time() - curr
  timers.append((i, curr, h))

for (i, curr, h) in timers:
  print(i, "AVEC", nbissues(h), "RESULTATS EN", curr, "SEC")

print("\n")


#TRAVAIL DES INSTANCES ==================================

setParameters(win=1, egalite=0, loss=0, scoref=0)

#AUTRES TESTS

#setup
ttab = [
[1, 2, 2, 2], 
[2, 2, 20, 2], 
[2, 2, 2, 2],
[2, 2, 2, 4]
]

live(GameInstance(random_tab(20, 20)))


#reperer le risque : faire la moyenne des valeurs du tableau, reperer la position dans les diagonales des valeurs les plus extremes, selon l'extremiter diviser à leur niveau la moyenne, pour plus de précision

#plus l'ecart est fort, plus on peut diviser (carte de division)