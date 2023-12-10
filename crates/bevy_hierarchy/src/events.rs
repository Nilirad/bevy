use bevy_ecs::{event::Event, prelude::Entity};

/// An [`Event`] sent each time the hierarchy is mutated.
///
/// [`Event`]: bevy_ecs::event::Event
#[derive(Event, Debug, Clone, PartialEq, Eq)]
pub enum HierarchyEvent {
    /// Sent whenever a parent-child relationship is set.
    ChildAdded {
        /// The child entity of the relationship.
        child: Entity,
        /// The parent entity of the relationship.
        parent: Entity,
    },
    /// Sent whenever a parent-child relationship is cleared.
    ChildRemoved {
        /// The child entity of the relationship.
        child: Entity,
        /// The parent entity of the relationship.
        parent: Entity,
    },
    /// Sent whenever a child is assigned to a new parent.
    ChildMoved {
        /// The child entity.
        child: Entity,
        /// The old parent.
        previous_parent: Entity,
        /// The new parent entity.
        new_parent: Entity,
    },
}
