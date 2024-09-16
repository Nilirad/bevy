//! The animation graph, which allows animations to be blended together.

use std::io::{self, Write};
use std::ops::{Index, IndexMut};

use bevy_asset::{io::Reader, Asset, AssetId, AssetLoader, AssetPath, Handle, LoadContext};
use bevy_reflect::{Reflect, ReflectSerialize};
use bevy_utils::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use ron::de::SpannedError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AnimationClip, AnimationTargetId};

/// A graph structure that describes how animation clips are to be blended
/// together.
///
/// Applications frequently want to be able to play multiple animations at once
/// and to fine-tune the influence that animations have on a skinned mesh. Bevy
/// uses an *animation graph* to store this information. Animation graphs are a
/// directed acyclic graph (DAG) that describes how animations are to be
/// weighted and combined together. Every frame, Bevy evaluates the graph from
/// the root and blends the animations together in a bottom-up fashion to
/// produce the final pose.
///
/// There are two types of nodes: *blend nodes* and *clip nodes*, both of which
/// can have an associated weight. Blend nodes have no associated animation clip
/// and simply affect the weights of all their descendant nodes. Clip nodes
/// specify an animation clip to play. When a graph is created, it starts with
/// only a single blend node, the root node.
///
/// For example, consider the following graph:
///
/// ```text
/// ┌────────────┐                                      
/// │            │                                      
/// │    Idle    ├─────────────────────┐                
/// │            │                     │                
/// └────────────┘                     │                
///                                    │                
/// ┌────────────┐                     │  ┌────────────┐
/// │            │                     │  │            │
/// │    Run     ├──┐                  ├──┤    Root    │
/// │            │  │  ┌────────────┐  │  │            │
/// └────────────┘  │  │   Blend    │  │  └────────────┘
///                 ├──┤            ├──┘                
/// ┌────────────┐  │  │    0.5     │                   
/// │            │  │  └────────────┘                   
/// │    Walk    ├──┘                                   
/// │            │                                      
/// └────────────┘                                      
/// ```
///
/// In this case, assuming that Idle, Run, and Walk are all playing with weight
/// 1.0, the Run and Walk animations will be equally blended together, then
/// their weights will be halved and finally blended with the Idle animation.
/// Thus the weight of Run and Walk are effectively half of the weight of Idle.
///
/// Nodes can optionally have a *mask*, a bitfield that restricts the set of
/// animation targets that the node and its descendants affect. Each bit in the
/// mask corresponds to a *mask group*, which is a set of animation targets
/// (bones). An animation target can belong to any number of mask groups within
/// the context of an animation graph.
///
/// When the appropriate bit is set in a node's mask, neither the node nor its
/// descendants will animate any animation targets belonging to that mask group.
/// That is, setting a mask bit to 1 *disables* the animation targets in that
/// group. If an animation target belongs to multiple mask groups, masking any
/// one of the mask groups that it belongs to will mask that animation target.
/// (Thus an animation target will only be animated if *all* of its mask groups
/// are unmasked.)
///
/// A common use of masks is to allow characters to hold objects. For this, the
/// typical workflow is to assign each character's hand to a mask group. Then,
/// when the character picks up an object, the application masks out the hand
/// that the object is held in for the character's animation set, then positions
/// the hand's digits as necessary to grasp the object. The character's
/// animations will continue to play but will not affect the hand, which will
/// continue to be depicted as holding the object.
///
/// Animation graphs are assets and can be serialized to and loaded from [RON]
/// files. Canonically, such files have an `.animgraph.ron` extension.
///
/// The animation graph implements [RFC 51]. See that document for more
/// information.
///
/// [RON]: https://github.com/ron-rs/ron
///
/// [RFC 51]: https://github.com/bevyengine/rfcs/blob/main/rfcs/51-animation-composition.md
#[derive(Asset, Reflect, Clone, Debug, Serialize)]
#[reflect(Serialize, Debug)]
#[serde(into = "SerializedAnimationGraph")]
pub struct AnimationGraph {
    /// The `petgraph` data structure that defines the animation graph.
    pub graph: AnimationDiGraph,

    /// The index of the root node in the animation graph.
    pub root: NodeIndex,

    /// The mask groups that each animation target (bone) belongs to.
    ///
    /// Each value in this map is a bitfield, in which 0 in bit position N
    /// indicates that the animation target doesn't belong to mask group N, and
    /// a 1 in position N indicates that the animation target does belong to
    /// mask group N.
    ///
    /// Animation targets not in this collection are treated as though they
    /// don't belong to any mask groups.
    pub mask_groups: HashMap<AnimationTargetId, AnimationMask>,
}

/// A type alias for the `petgraph` data structure that defines the animation
/// graph.
pub type AnimationDiGraph = DiGraph<AnimationGraphNode, (), u32>;

/// The index of either an animation or blend node in the animation graph.
///
/// These indices are the way that [`crate::AnimationPlayer`]s identify
/// particular animations.
pub type AnimationNodeIndex = NodeIndex<u32>;

/// An individual node within an animation graph.
///
/// If `clip` is present, this is a *clip node*. Otherwise, it's a *blend node*.
/// Both clip and blend nodes can have weights, and those weights are propagated
/// down to descendants.
#[derive(Clone, Reflect, Debug)]
pub struct AnimationGraphNode {
    /// The animation clip associated with this node, if any.
    ///
    /// If the clip is present, this node is an *animation clip node*.
    /// Otherwise, this node is a *blend node*.
    pub clip: Option<Handle<AnimationClip>>,

    /// A bitfield specifying the mask groups that this node and its descendants
    /// will not affect.
    ///
    /// A 0 in bit N indicates that this node and its descendants *can* animate
    /// animation targets in mask group N, while a 1 in bit N indicates that
    /// this node and its descendants *cannot* animate mask group N.
    pub mask: AnimationMask,

    /// The weight of this node.
    ///
    /// Weights are propagated down to descendants. Thus if an animation clip
    /// has weight 0.3 and its parent blend node has weight 0.6, the computed
    /// weight of the animation clip is 0.18.
    pub weight: f32,
}

/// An [`AssetLoader`] that can load [`AnimationGraph`]s as assets.
///
/// The canonical extension for [`AnimationGraph`]s is `.animgraph.ron`. Plain
/// `.animgraph` is supported as well.
#[derive(Default)]
pub struct AnimationGraphAssetLoader;

/// Various errors that can occur when serializing or deserializing animation
/// graphs to and from RON, respectively.
#[derive(Error, Debug)]
pub enum AnimationGraphLoadError {
    /// An I/O error occurred.
    #[error("I/O")]
    Io(#[from] io::Error),
    /// An error occurred in RON serialization or deserialization.
    #[error("RON serialization")]
    Ron(#[from] ron::Error),
    /// An error occurred in RON deserialization, and the location of the error
    /// is supplied.
    #[error("RON serialization")]
    SpannedRon(#[from] SpannedError),
}

/// A version of [`AnimationGraph`] suitable for serializing as an asset.
///
/// Animation nodes can refer to external animation clips, and the [`AssetId`]
/// is typically not sufficient to identify the clips, since the
/// [`bevy_asset::AssetServer`] assigns IDs in unpredictable ways. That fact
/// motivates this type, which replaces the `Handle<AnimationClip>` with an
/// asset path.  Loading an animation graph via the [`bevy_asset::AssetServer`]
/// actually loads a serialized instance of this type, as does serializing an
/// [`AnimationGraph`] through `serde`.
#[derive(Serialize, Deserialize)]
pub struct SerializedAnimationGraph {
    /// Corresponds to the `graph` field on [`AnimationGraph`].
    pub graph: DiGraph<SerializedAnimationGraphNode, (), u32>,
    /// Corresponds to the `root` field on [`AnimationGraph`].
    pub root: NodeIndex,
    /// Corresponds to the `mask_groups` field on [`AnimationGraph`].
    pub mask_groups: HashMap<AnimationTargetId, AnimationMask>,
}

/// A version of [`AnimationGraphNode`] suitable for serializing as an asset.
///
/// See the comments in [`SerializedAnimationGraph`] for more information.
#[derive(Serialize, Deserialize)]
pub struct SerializedAnimationGraphNode {
    /// Corresponds to the `clip` field on [`AnimationGraphNode`].
    pub clip: Option<SerializedAnimationClip>,
    /// Corresponds to the `mask` field on [`AnimationGraphNode`].
    pub mask: AnimationMask,
    /// Corresponds to the `weight` field on [`AnimationGraphNode`].
    pub weight: f32,
}

/// A version of `Handle<AnimationClip>` suitable for serializing as an asset.
///
/// This replaces any handle that has a path with an [`AssetPath`]. Failing
/// that, the asset ID is serialized directly.
#[derive(Serialize, Deserialize)]
pub enum SerializedAnimationClip {
    /// Records an asset path.
    AssetPath(AssetPath<'static>),
    /// The fallback that records an asset ID.
    ///
    /// Because asset IDs can change, this should not be relied upon. Prefer to
    /// use asset paths where possible.
    AssetId(AssetId<AnimationClip>),
}

/// The type of an animation mask bitfield.
///
/// Bit N corresponds to mask group N.
///
/// Because this is a 64-bit value, there is currently a limitation of 64 mask
/// groups per animation graph.
pub type AnimationMask = u64;

impl AnimationGraph {
    /// Creates a new animation graph with a root node and no other nodes.
    pub fn new() -> Self {
        let mut graph = DiGraph::default();
        let root = graph.add_node(AnimationGraphNode::default());
        Self {
            graph,
            root,
            mask_groups: HashMap::new(),
        }
    }

    /// A convenience function for creating an [`AnimationGraph`] from a single
    /// [`AnimationClip`].
    ///
    /// The clip will be a direct child of the root with weight 1.0. Both the
    /// graph and the index of the added node are returned as a tuple.
    pub fn from_clip(clip: Handle<AnimationClip>) -> (Self, AnimationNodeIndex) {
        let mut graph = Self::new();
        let node_index = graph.add_clip(clip, 1.0, graph.root);
        (graph, node_index)
    }

    /// A convenience method to create an [`AnimationGraph`]s with an iterator
    /// of clips.
    ///
    /// All of the animation clips will be direct children of the root with
    /// weight 1.0.
    ///
    /// Returns the the graph and indices of the new nodes.
    pub fn from_clips<'a, I>(clips: I) -> (Self, Vec<AnimationNodeIndex>)
    where
        I: IntoIterator<Item = Handle<AnimationClip>>,
        <I as IntoIterator>::IntoIter: 'a,
    {
        let mut graph = Self::new();
        let indices = graph.add_clips(clips, 1.0, graph.root).collect();
        (graph, indices)
    }

    /// Adds an [`AnimationClip`] to the animation graph with the given weight
    /// and returns its index.
    ///
    /// The animation clip will be the child of the given parent. The resulting
    /// node will have no mask.
    pub fn add_clip(
        &mut self,
        clip: Handle<AnimationClip>,
        weight: f32,
        parent: AnimationNodeIndex,
    ) -> AnimationNodeIndex {
        let node_index = self.graph.add_node(AnimationGraphNode {
            clip: Some(clip),
            mask: 0,
            weight,
        });
        self.graph.add_edge(parent, node_index, ());
        node_index
    }

    /// Adds an [`AnimationClip`] to the animation graph with the given weight
    /// and mask, and returns its index.
    ///
    /// The animation clip will be the child of the given parent.
    pub fn add_clip_with_mask(
        &mut self,
        clip: Handle<AnimationClip>,
        mask: AnimationMask,
        weight: f32,
        parent: AnimationNodeIndex,
    ) -> AnimationNodeIndex {
        let node_index = self.graph.add_node(AnimationGraphNode {
            clip: Some(clip),
            mask,
            weight,
        });
        self.graph.add_edge(parent, node_index, ());
        node_index
    }

    /// A convenience method to add multiple [`AnimationClip`]s to the animation
    /// graph.
    ///
    /// All of the animation clips will have the same weight and will be
    /// parented to the same node.
    ///
    /// Returns the indices of the new nodes.
    pub fn add_clips<'a, I>(
        &'a mut self,
        clips: I,
        weight: f32,
        parent: AnimationNodeIndex,
    ) -> impl Iterator<Item = AnimationNodeIndex> + 'a
    where
        I: IntoIterator<Item = Handle<AnimationClip>>,
        <I as IntoIterator>::IntoIter: 'a,
    {
        clips
            .into_iter()
            .map(move |clip| self.add_clip(clip, weight, parent))
    }

    /// Adds a blend node to the animation graph with the given weight and
    /// returns its index.
    ///
    /// The blend node will be placed under the supplied `parent` node. During
    /// animation evaluation, the descendants of this blend node will have their
    /// weights multiplied by the weight of the blend. The blend node will have
    /// no mask.
    pub fn add_blend(&mut self, weight: f32, parent: AnimationNodeIndex) -> AnimationNodeIndex {
        let node_index = self.graph.add_node(AnimationGraphNode {
            clip: None,
            mask: 0,
            weight,
        });
        self.graph.add_edge(parent, node_index, ());
        node_index
    }

    /// Adds a blend node to the animation graph with the given weight and
    /// returns its index.
    ///
    /// The blend node will be placed under the supplied `parent` node. During
    /// animation evaluation, the descendants of this blend node will have their
    /// weights multiplied by the weight of the blend. Neither this node nor its
    /// descendants will affect animation targets that belong to mask groups not
    /// in the given `mask`.
    pub fn add_blend_with_mask(
        &mut self,
        mask: AnimationMask,
        weight: f32,
        parent: AnimationNodeIndex,
    ) -> AnimationNodeIndex {
        let node_index = self.graph.add_node(AnimationGraphNode {
            clip: None,
            mask,
            weight,
        });
        self.graph.add_edge(parent, node_index, ());
        node_index
    }

    /// Adds an edge from the edge `from` to `to`, making `to` a child of
    /// `from`.
    ///
    /// The behavior is unspecified if adding this produces a cycle in the
    /// graph.
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, ());
    }

    /// Removes an edge between `from` and `to` if it exists.
    ///
    /// Returns true if the edge was successfully removed or false if no such
    /// edge existed.
    pub fn remove_edge(&mut self, from: NodeIndex, to: NodeIndex) -> bool {
        self.graph
            .find_edge(from, to)
            .map(|edge| self.graph.remove_edge(edge))
            .is_some()
    }

    /// Returns the [`AnimationGraphNode`] associated with the given index.
    ///
    /// If no node with the given index exists, returns `None`.
    pub fn get(&self, animation: AnimationNodeIndex) -> Option<&AnimationGraphNode> {
        self.graph.node_weight(animation)
    }

    /// Returns a mutable reference to the [`AnimationGraphNode`] associated
    /// with the given index.
    ///
    /// If no node with the given index exists, returns `None`.
    pub fn get_mut(&mut self, animation: AnimationNodeIndex) -> Option<&mut AnimationGraphNode> {
        self.graph.node_weight_mut(animation)
    }

    /// Returns an iterator over the [`AnimationGraphNode`]s in this graph.
    pub fn nodes(&self) -> impl Iterator<Item = AnimationNodeIndex> {
        self.graph.node_indices()
    }

    /// Serializes the animation graph to the given [`Write`]r in RON format.
    ///
    /// If writing to a file, it can later be loaded with the
    /// [`AnimationGraphAssetLoader`] to reconstruct the graph.
    pub fn save<W>(&self, writer: &mut W) -> Result<(), AnimationGraphLoadError>
    where
        W: Write,
    {
        let mut ron_serializer = ron::ser::Serializer::new(writer, None)?;
        Ok(self.serialize(&mut ron_serializer)?)
    }

    /// Adds an animation target (bone) to the mask group with the given ID.
    ///
    /// Calling this method multiple times with the same animation target but
    /// different mask groups will result in that target being added to all of
    /// the specified groups.
    pub fn add_target_to_mask_group(&mut self, target: AnimationTargetId, mask_group: u32) {
        *self.mask_groups.entry(target).or_default() |= 1 << mask_group;
    }
}

impl AnimationGraphNode {
    /// Masks out the mask groups specified by the given `mask` bitfield.
    ///
    /// A 1 in bit position N causes this function to mask out mask group N, and
    /// thus neither this node nor its descendants will animate any animation
    /// targets that belong to group N.
    pub fn add_mask(&mut self, mask: AnimationMask) -> &mut Self {
        self.mask |= mask;
        self
    }

    /// Unmasks the mask groups specified by the given `mask` bitfield.
    ///
    /// A 1 in bit position N causes this function to unmask mask group N, and
    /// thus this node and its descendants will be allowed to animate animation
    /// targets that belong to group N, unless another mask masks those targets
    /// out.
    pub fn remove_mask(&mut self, mask: AnimationMask) -> &mut Self {
        self.mask &= !mask;
        self
    }

    /// Masks out the single mask group specified by `group`.
    ///
    /// After calling this function, neither this node nor its descendants will
    /// animate any animation targets that belong to the given `group`.
    pub fn add_mask_group(&mut self, group: u32) -> &mut Self {
        self.add_mask(1 << group)
    }

    /// Unmasks the single mask group specified by `group`.
    ///
    /// After calling this function, this node and its descendants will be
    /// allowed to animate animation targets that belong to the given `group`,
    /// unless another mask masks those targets out.
    pub fn remove_mask_group(&mut self, group: u32) -> &mut Self {
        self.remove_mask(1 << group)
    }
}

impl Index<AnimationNodeIndex> for AnimationGraph {
    type Output = AnimationGraphNode;

    fn index(&self, index: AnimationNodeIndex) -> &Self::Output {
        &self.graph[index]
    }
}

impl IndexMut<AnimationNodeIndex> for AnimationGraph {
    fn index_mut(&mut self, index: AnimationNodeIndex) -> &mut Self::Output {
        &mut self.graph[index]
    }
}

impl Default for AnimationGraphNode {
    fn default() -> Self {
        Self {
            clip: None,
            mask: 0,
            weight: 1.0,
        }
    }
}

impl Default for AnimationGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetLoader for AnimationGraphAssetLoader {
    type Asset = AnimationGraph;

    type Settings = ();

    type Error = AnimationGraphLoadError;

    async fn load<'a>(
        &'a self,
        reader: &'a mut dyn Reader,
        _: &'a Self::Settings,
        load_context: &'a mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        // Deserialize a `SerializedAnimationGraph` directly, so that we can
        // get the list of the animation clips it refers to and load them.
        let mut deserializer = ron::de::Deserializer::from_bytes(&bytes)?;
        let serialized_animation_graph = SerializedAnimationGraph::deserialize(&mut deserializer)
            .map_err(|err| deserializer.span_error(err))?;

        // Load all `AssetPath`s to convert from a
        // `SerializedAnimationGraph` to a real `AnimationGraph`.
        Ok(AnimationGraph {
            graph: serialized_animation_graph.graph.map(
                |_, serialized_node| AnimationGraphNode {
                    clip: serialized_node.clip.as_ref().map(|clip| match clip {
                        SerializedAnimationClip::AssetId(asset_id) => Handle::Weak(*asset_id),
                        SerializedAnimationClip::AssetPath(asset_path) => {
                            load_context.load(asset_path)
                        }
                    }),
                    mask: serialized_node.mask,
                    weight: serialized_node.weight,
                },
                |_, _| (),
            ),
            root: serialized_animation_graph.root,
            mask_groups: serialized_animation_graph.mask_groups,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["animgraph", "animgraph.ron"]
    }
}

impl From<AnimationGraph> for SerializedAnimationGraph {
    fn from(animation_graph: AnimationGraph) -> Self {
        // If any of the animation clips have paths, then serialize them as
        // `SerializedAnimationClip::AssetPath` so that the
        // `AnimationGraphAssetLoader` can load them.
        Self {
            graph: animation_graph.graph.map(
                |_, node| SerializedAnimationGraphNode {
                    weight: node.weight,
                    mask: node.mask,
                    clip: node.clip.as_ref().map(|clip| match clip.path() {
                        Some(path) => SerializedAnimationClip::AssetPath(path.clone()),
                        None => SerializedAnimationClip::AssetId(clip.id()),
                    }),
                },
                |_, _| (),
            ),
            root: animation_graph.root,
            mask_groups: animation_graph.mask_groups,
        }
    }
}